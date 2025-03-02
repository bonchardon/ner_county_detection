from asyncio import run

from loguru import logger

from core.ai.ner_recognizer.llm_builder import ModelBuilder
from core.ai.rag_llama.rag_builder import BuildRAG


async def main() -> None:
    # data: list[str] = await CollectData().json_file()
    # logger.info(data.__len__())
    # preprocessed_data: list[DataSet] | None = await PreprocessingFormula().preprocessing_pipeline(data_set=data)
    # logger.info(preprocessed_data[:5])

    # model_check = await ModelBuilder().llm_builder()
    # logger.info(model_check)
    # return

    return await BuildRAG().embedding_data()

if __name__ == '__main__':
    run(main())
