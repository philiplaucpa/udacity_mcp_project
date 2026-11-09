import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, List, Dict, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import re

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "mcpServers" not in config:
                raise ValueError("Configuration file is missing required 'mcpServers' field")

            return config

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {file_path}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid configuration in {file_path}: {e}")
            raise

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        # complete params
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ,** self.config["env"]} if self.config.get("env") else None,
            )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()

        tools: List[ToolDefinition] = []
        for tool in tools_response.tools:
            tool_def: ToolDefinition = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            tools.append(tool_def)

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(seconds=60),
                )
                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool {tool_name}: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"Max retries reached for {tool_name}. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""
    
    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client
        
    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:
            
            await self.sqlite_server.execute_tool("write_query", {
                "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                    features TEXT,  -- JSON array
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            })
            
            logging.info("✓ Data extraction tables initialized")
            
        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            response = self.anthropic.messages.create(
                max_tokens=1024,
                model='claude-sonnet-5-5',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            text_content = ""
            for content in response.content:
                if content.type == 'text':
                    text_content += content.text
            
            return text_content.strip()
            
        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'
    
    async def extract_and_store_data(self, user_query: str, llm_response: str, 
                                   source_url: str = None) -> None:
        """Extract structured data from LLM response and store it."""
        try:            
            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format:
            
            Text: {llm_response}
            
            Extract pricing plans with this structure:
            {{
                "company_name": "company name",
                "plans": [
                    {{
                        "plan_name": "plan name",
                        "input_tokens": number or null,
                        "output_tokens": number or null,
                        "currency": "USD",
                        "billing_period": "monthly/yearly/one-time",
                        "features": ["feature1", "feature2"],
                        "limitations": "any limitations mentioned",
                        "query": "the user's query"
                    }}
                ]
            }}
            
            Return only valid JSON, no other text. Do not return your response enclosed in ```json```
            """
            
            extraction_response = await self._get_structured_extraction(extraction_prompt)
            extraction_response = extraction_response.replace("```json\n", "").replace("```", "")
            pricing_data = json.loads(extraction_response)
            
            # Escape single quotes so text like "user's query" doesn't break the SQL
            def esc(value: Any) -> str:
                return str(value).replace("'", "''")

            for plan in pricing_data.get("plans", []):
                await self.sqlite_server.execute_tool("write_query", {
                    "query": f"""
                    INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
                    VALUES (
                        '{esc(pricing_data.get("company_name", "Unknown"))}',
                        '{esc(plan.get("plan_name", "Unknown Plan"))}',
                        '{esc(plan.get("input_tokens") or 0)}',
                        '{esc(plan.get("output_tokens") or 0)}',
                        '{esc(plan.get("currency", "USD"))}',
                        '{esc(plan.get("billing_period", "unknown"))}',
                        '{esc(json.dumps(plan.get("features", [])))}',
                        '{esc(plan.get("limitations", ""))}',
                        '{esc(user_query)}'
                    )
                    """
                })

            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans")
            
        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(api_key=api_key)
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        messages = [{'role': 'user', 'content': query}]
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model='claude-sonnet-5-5',
            tools=self.available_tools,
            messages=messages
        )

        full_response = ""
        source_url = None
        used_web_search = False

        process_query = True
        while process_query:
            assistant_content = []
            tool_results = []
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    full_response += content.text + "\n"
                    assistant_content.append(content)

                elif content.type == 'tool_use':
                    # 1. Keep the tool request in the assistant's message
                    assistant_content.append(content)

                    # 2. Get the tool details
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name
                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # 3. Find the server that has this tool
                    server_name = self.tool_to_server.get(tool_name)
                    server = next((s for s in self.servers if s.name == server_name), None)

                    # 4. Run the tool on that server
                    is_error = False
                    if server is None:
                        result_text = f"Tool '{tool_name}' not found on any server"
                        is_error = True
                    else:
                        try:
                            result = await server.execute_tool(tool_name, tool_args)
                            result_text = "\n".join(
                                getattr(item, "text", str(item)) for item in result.content
                            )
                            is_error = bool(getattr(result, "isError", False))
                        except Exception as e:
                            result_text = f"Error executing tool '{tool_name}': {e}"
                            is_error = True

                    # Remember where the data came from (used when storing pricing data)
                    if tool_name == "scrape_websites":
                        used_web_search = True
                    if not source_url:
                        source_url = self._extract_url_from_result(result_text)

                    # 5. Collect the tool result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                        "is_error": is_error,
                    })

            if tool_results:
                # Send the assistant's tool requests and all their results back to Claude
                messages.append({'role': 'assistant', 'content': assistant_content})
                messages.append({'role': 'user', 'content': tool_results})

                # 6. Ask Claude again with the tool results
                response = self.anthropic.messages.create(
                    max_tokens=2024,
                    model='claude-sonnet-5-5',
                    tools=self.available_tools,
                    messages=messages
                )
            else:
                # 7. No tool calls means Claude is done
                process_query = False

        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(query, full_response.strip(), source_url)

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'show data':
                    await self.show_stored_data()
                    continue
                    
                await self.process_query(query)
                print("\n")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return
            
        try:
            pricing = await self.sqlite_server.execute_tool("read_query", {
                "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency "
                         "FROM pricing_plans ORDER BY created_at DESC LIMIT 5"
            })

            print("\nRecently Stored Data:")
            print("=" * 50)

            print("\nPricing Plans:")
            # The SQLite server returns rows as text. It may be JSON or a Python-style
            # list of dicts (single quotes), so fall back to ast.literal_eval.
            raw_text = pricing.content[0].text if pricing.content else "[]"
            try:
                rows = json.loads(raw_text)
            except json.JSONDecodeError:
                import ast
                rows = ast.literal_eval(raw_text)

            if not rows:
                print("  (no pricing plans stored yet)")
            for plan in rows:
                print(
                    f"  • {plan['company_name']}: {plan['plan_name']} - "
                    f"Input Token ${plan['input_tokens']}, "
                    f"Output Token ${plan['output_tokens']} ({plan['currency']})"
                )

            print("=" * 50)
        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")
            
            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    
    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"
    
    server_config = config.load_config(config_file)
    
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())