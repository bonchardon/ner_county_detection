from asyncio import run

from loguru import logger

from core.train_test_set.collect_data import CollectData
from core.train_test_set.preprocess_part import PreprocessingFormula
from train_test_set.corpus import DataSet


async def main() -> None:
    data: list[str] = await CollectData().json_file()
    logger.info(data.__len__())
    preprocessed_data: list[DataSet] | None = await PreprocessingFormula().preprocessing_pipeline(data_set=data)
    logger.info(preprocessed_data[:5])
    return

if __name__ == '__main__':
    run(main())
