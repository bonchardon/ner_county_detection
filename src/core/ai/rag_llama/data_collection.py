from typing import Any

from time import sleep
from helium import start_chrome, find_all, S
from loguru import logger

from core.ai.rag_llama import consts


class DataCollector:
    url: str = consts.DATA_LINK
    driver = start_chrome(url, headless=True)

    def fetch_links_with_helium(self) -> list[str]:
        """Fetch page using Helium and extract all article links within li tags."""
        sleep(2)
        links: list = []
        ul_elements: list = find_all(S('ul'))

        for ul in ul_elements:
            try:
                ul_html: Any = ul.web_element.get_attribute('outerHTML')  # Retrieve actual web element and get outerHTML
                if 'data-unit-page' in ul_html and 'ILERn' in ul_html:
                    li_elements: Any = ul.find_all('li')

                    for li in li_elements:
                        try:
                            link: Any = li.find_all('a')
                            if link and link[0].get_attribute('href'):
                                href = link[0].get_attribute('href')
                                links.append(href)
                        except Exception as e:
                            logger.warning(f"Failed to extract link from <li>: {e}")
            except Exception as e:
                logger.warning(f"Error with <ul> element: {e}")

        return links
