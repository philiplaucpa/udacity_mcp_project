
import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    app = FirecrawlApp(api_key=api_key)
    
    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    # {
    #     "cloudrift_ai": {
    #         "provider_name": "cloudrift_ai",
    #         "url": "https://www.cloudrift.ai/inference",
    #         "domain": "www.cloudrift.ai",
    #         "scraped_at": "2025-10-23T00:44:59.902569",
    #         "formats": [
    #             "markdown",
    #             "html"
    #         ],
    #         "success": "true",
    #         "content_files": {
    #             "markdown": "cloudrift_ai_markdown.txt",
    #             "html": "cloudrift_ai_html.txt"
    #         },
    #         "title": "AI Inference",
    #         "description": "Scraped content goes here"
    #     }
    # }
    metadata_file = os.path.join(path, "scraped_metadata.json")

    # 1. Load existing metadata (or start empty)
    scraped_metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                scraped_metadata = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            scraped_metadata = {}

    # 2. Track successful scrapes
    successful_scrapes = []

    # 3. Loop through websites
    for provider_name, url in websites.items():
        try:
            logger.info(f"Scraping {provider_name}: {url}")

            scrape_result = app.scrape(url, formats=formats).model_dump()

            metadata = {
                "provider_name": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": False,
                "content_files": {},
                "title": "",
                "description": "",
            }

            # Newer Firecrawl SDKs return the document directly (no 'success' key),
            # so also treat the presence of content as success.
            has_content = any(scrape_result.get(fmt) for fmt in formats)
            if scrape_result.get("success", False) or has_content:
                metadata["success"] = True

                # Save each requested format to its own file
                for format_type in formats:
                    content = scrape_result.get(format_type)
                    if content:
                        filename = f"{provider_name}_{format_type}.txt"
                        with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
                            f.write(content)
                        metadata["content_files"][format_type] = filename

                # Title / description live in the page metadata
                page_meta = scrape_result.get("metadata") or {}
                metadata["title"] = page_meta.get("title", "") or ""
                metadata["description"] = page_meta.get("description", "") or ""

                successful_scrapes.append(provider_name)
                logger.info(f"Successfully scraped {provider_name}")
            else:
                logger.error(f"Failed to scrape {provider_name}: {scrape_result.get('error', 'Unknown error')}")

            scraped_metadata[provider_name] = metadata

        except Exception as e:
            logger.error(f"Error scraping {provider_name} ({url}): {e}")
            scraped_metadata[provider_name] = {
                "provider_name": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": False,
                "error": str(e),
            }

    # 5. Save the full metadata back to disk
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(scraped_metadata, f, indent=4)

    logger.info(f"Scraping complete. {len(successful_scrapes)}/{len(websites)} succeeded: {successful_scrapes}")
    return successful_scrapes

@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    logger.info(f"Checking metadata file: {metadata_file}")

    try:
        # 1. Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            scraped_metadata = json.load(f)

        # 2. Find a match by provider name, URL, or domain
        for provider_name, metadata in scraped_metadata.items():
            if identifier in (provider_name, metadata.get("url", ""), metadata.get("domain", "")):
                # 3. Build the result with file contents
                result = metadata.copy()
                if "content_files" in metadata:
                    result["content"] = {}
                    for format_type, filename in metadata["content_files"].items():
                        with open(os.path.join(SCRAPE_DIR, filename), "r", encoding="utf-8") as f:
                            result["content"][format_type] = f.read()
                return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error extracting info for {identifier}: {e}")

    # 4. No match or error
    return f"There's no saved information related to identifier '{identifier}'."

if __name__ == "__main__":
    mcp.run(transport="stdio")