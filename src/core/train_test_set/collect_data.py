# todo:
#  1) find viable and comprehensive data sources; (what is the main topic, I guess NER to identify countries, among )
#  2) preprocess data for further use;
#  3) apply ETL approach

from json import dump, load

from loguru import logger

from core.train_test_set.consts import FILE_1_PATH


class CollectData:
    async def json_file(self) -> list[str] | None:
        with open(FILE_1_PATH, 'r') as file:
            json_file: list[dict[str, str]] = load(file)
        if not (json_data := [data.get('title_p') for data in json_file]):
            logger.error('Having issues when parsing data.')
            return
        logger.success(f'Data has been retrieved successfully. Here is some part of it: {json_data[:3]}')
        return json_data

    async def japanese_web_scrapping(self):
        ...

    async def all_data_retrieved(self):
        # todo: combine all data

        ...



