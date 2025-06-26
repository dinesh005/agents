import os
import getpass
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain_tavily import TavilyExtract
from langchain_tavily import TavilyCrawl


load_dotenv()

class TavilyTools:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        self.client = TavilyClient(api_key=self.api_key)
        self.search_tool = TavilySearch(max_results=5, topic="general")
        self.extract_tool = TavilyExtract(extract_depth="basic",include_images=False)
        self.crawl_tool = TavilyCrawl()

    def search(self, query: str):
        return self.search_tool.run(query)

    def extract(self, query: str):
        return self.extract_tool.run(query)

    def crawl(self, url: str):
        return self.crawl_tool.run(url)
