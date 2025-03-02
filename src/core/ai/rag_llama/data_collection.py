# from asyncio import run, sleep
#
# from httpx import AsyncClient, Response
# from selectolax.parser import HTMLParser, Node
#
# from loguru import logger
#
# from core.ai.rag_llama import consts
#
#
# class DataCollector:
#
#     def __init__(self):
#         self.client = AsyncClient()
#
#     async def fetch_links(self) -> str | None:
#         if not (response := await self.client.get(consts.MAIN_LINK, headers=consts.HEADERS)):
#             logger.warning('Having an error when parsing data from a webpage.')
#             return
#         return response.text
#
#     async def collect_all_links(self):
#         tree = HTMLParser(await self.fetch_links())
#         links = []
#         for a_tag in tree.css('a[href]'):
#             href = a_tag.attributes.get('href')
#             if href and href.startswith('/articles/'):
#                 links.append(href)
#         logger.info(links)
#         return links
#
#     async def final_data(self):
#         return await self.collect_all_links()
#
#
# async def main():
#     return await DataCollector().final_data()
#
# if __name__ == '__main__':
#     run(main())

from time import sleep
from helium import start_chrome, find_all, S
from loguru import logger


class DataCollector:

    def __init__(self):
        # URL to start scraping
        self.url = "https://www.asahi.com/topics/AP-7274059d-8405-4d7f-8dbd-8203b01bbbc8/timeline-unit/735073dc-92bf-485a-909c-0d823315cb69/?iref=com_topics_7274059d-8405-4d7f-8dbd-8203b01bbbc8_timeline_readmore"
        self.driver = start_chrome(self.url, headless=True)

    def fetch_links_with_helium(self) -> list[str]:
        """Fetch page using Helium and extract all article links within li tags."""
        sleep(2)
        links = []

        # Find all <ul> elements using the S selector
        ul_elements = find_all(S('ul'))

        for ul in ul_elements:
            try:
                # Check if the specific attributes are in the 'outerHTML' of the <ul> element
                ul_html = ul.web_element.get_attribute('outerHTML')  # Retrieve actual web element and get outerHTML
                if 'data-unit-page' in ul_html and 'ILERn' in ul_html:
                    li_elements = ul.find_all('li')

                    for li in li_elements:
                        try:
                            # Find <a> tags inside <li> elements
                            link = li.find_all('a')
                            if link and link[0].get_attribute('href'):
                                href = link[0].get_attribute('href')
                                links.append(href)
                        except Exception as e:
                            logger.warning(f"Failed to extract link from <li>: {e}")
            except Exception as e:
                logger.warning(f"Error with <ul> element: {e}")

        return links

    def collect_all_links(self):
        links = self.fetch_links_with_helium()
        logger.info(f"Found {len(links)} article links.")
        return links

    def final_data(self):
        return self.collect_all_links()


if __name__ == '__main__':
    collector = DataCollector()
    links = collector.final_data()
    print(links)

