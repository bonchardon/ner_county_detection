from asyncio import run

from loguru import logger

from train_test_set.corpus import DataSet

from core.ai.rag_llama.rag_builder import BuildRAG
from core.train_test_set.preprocess_part import PreprocessingFormula
from core.ai.ner_recognizer.llm_builder import ModelBuilder


async def main() -> None:
    trial_sentence = ['Mickie3777 日本は「半島人」アレルギーがあるのと、もう一つは今のクルド人の様な難民との区別がつかない人が多いんですよ。だからウクライナからの一時的な戦時避難民に対してさえ冷たい声が上がっていました。']
    preprocessed_data: list[DataSet] | None = await PreprocessingFormula().preprocessing_pipeline(
        data_set=trial_sentence
    )
    logger.info(preprocessed_data[:5])
    return await ModelBuilder().prompt_ner(input_text=preprocessed_data)

if __name__ == '__main__':
    run(main())
