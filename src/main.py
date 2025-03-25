from json import dump, load
from asyncio import run, gather

from loguru import logger

from core.train_test_set.corpus import DataSet

from core.ai.ner_recognizer.llm_builder import ModelBuilder
from core.train_test_set.preprocess_part import PreprocessingFormula


async def process_item(item):
    logger.info(f"Processing: {item['title_p']}")
    preprocessed_data = await PreprocessingFormula().preprocessing_pipeline(data_set=[item['title_p']])
    result = await ModelBuilder().prompt_ner(input_text=item['title_p'])
    item['maryna_s_reply'] = result
    logger.info(f'NER result: {result}')
    return item

async def main() -> None:
    with open('src/data/processed_data.json', 'r', encoding='utf-8') as f:
        trial_reply = load(f)
    processed = await gather(*[process_item(item) for item in trial_reply])
    with open('src/data/processed_data_ner_task.json', 'w', encoding='utf-8') as f:
        dump(processed, f, ensure_ascii=False, indent=2)
    logger.info('NER processing complete.')

if __name__ == '__main__':
    run(main())
