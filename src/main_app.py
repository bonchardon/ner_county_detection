from json import dump, load
from asyncio import run, gather

from uvicorn import run

from loguru import logger

from fastapi import FastAPI

# from core.train_test_set.corpus import DataSet

from core.ai.ner_recognizer.llm_builder import ModelBuilder
from core.train_test_set.preprocess_part import PreprocessingFormula


app: FastAPI = FastAPI()

@app.get('/')
async def root():
    return 'API is running and ready to betested.'

@app.post('/process/')
async def main(request_data):
    logger.info(f"Processing: {request_data}")
    await PreprocessingFormula().preprocessing_pipeline(data_set=request_data)
    result = await ModelBuilder().prompt_ner(input_text=request_data)
    logger.info(f'NER result: {result}')
    return {
            "title_p": request_data,
            "maryna_s_reply": result
        }

if __name__ == '__main__':
    run('main_app:app', host='0.0.0.0', port=8000, reload=True)