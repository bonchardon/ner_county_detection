from json import dump, load
from asyncio import run, gather

from loguru import logger

from fastapi import FastAPI

# from core.train_test_set.corpus import DataSet

from core.ai.ner_recognizer.llm_builder import ModelBuilder
from core.train_test_set.preprocess_part import PreprocessingFormula


async def process_item():
    trial_sentence = 'Chinaによるチベットやウイグル弾圧は国外脱出が命懸け。脱出できても家族を人質に脅迫される。ロヒンギャ難民やシリア難民は、距離という物理的な問題で日本は選択肢から外れる。ウクライナ戦争難民は女性と子供。日本で難民申請してる大半は、なぜか査証免除やら、飛行機で来日したヒトなんだよ。'
    # if not (preprocessed_data := await PreprocessingFormula().preprocessing_pipeline(data_set=trial_sentence)):
    #     logger.warning('Cannot go through preprocess part.')
    #     return
    result: ModelBuilder = await ModelBuilder().prompt_ner(input_text=trial_sentence)
    logger.info(result)
    # item['maryna_s_reply'] = result
    # logger.info(f'NER result: {result}')
    # return item

async def main() -> None:
    with open('src/data/processed_data.json', 'r', encoding='utf-8') as f:
        trial_reply = load(f)
    processed: list = await gather(*[process_item(item) for item in trial_reply])
    logger.info(processed)
    # with open('src/data/processed_data_ner_task.json', 'w', encoding='utf-8') as f:
    #     dump(processed, f, ensure_ascii=False, indent=2)
    logger.info('NER processing complete.')


if __name__ == '__main__':
    run(process_item())
