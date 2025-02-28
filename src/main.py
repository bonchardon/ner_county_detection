from asyncio import run

from loguru import logger

from core.ai.ner_recognizer.llm_builder import ModelBuilder


async def main() -> None:
    # data: list[str] = await CollectData().json_file()
    # logger.info(data.__len__())
    # preprocessed_data: list[DataSet] | None = await PreprocessingFormula().preprocessing_pipeline(data_set=data)
    # logger.info(preprocessed_data[:5])

    model_check = await ModelBuilder().llm_builder()
    logger.info(model_check)
    return

if __name__ == '__main__':
    run(main())
#
# import torch
# from loguru import logger
# from transformers import pipeline, AutoTokenizer, PreTrainedTokenizerFast, AutoModelForTokenClassification
# from langchain import HuggingFacePipeline
# from typing import Any, List
#
# from core.ai.enums import AiModels
# from core.ai.models import JapaneseNamedEntitiesIdentificator, CountriesReCheck
#
#
# class ModelBuilder:
#     @staticmethod
#     def create_ner_pipeline(model_name: str):
#         """Creates and returns a NER pipeline for country identification."""
#         model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch.float16,
#                                                                 trust_remote_code=True)
#         tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
#
#         # Set up the NER pipeline
#         ner_pipeline = pipeline(
#             task='ner',
#             model=model,
#             tokenizer=tokenizer,
#             aggregation_strategy="simple"  # Optional: aggregate multi-token entities into one
#         )
#         return ner_pipeline
#
#     @classmethod
#     async def llm_builder(cls, input_text: str) -> List[dict]:
#         """Fetches countries from the input text using NER model."""
#         ner_pipeline = cls.create_ner_pipeline(AiModels.LLM_MODEL_3_1_SWALLOW)
#
#         # Run the NER pipeline on the input text
#         result = ner_pipeline(input_text)
#
#         # Log and return the result
#         logger.info(f"NER Result: {result}")
#         return result
#
#     @classmethod
#     async def country_identifier(cls,
#                                  response: JapaneseNamedEntitiesIdentificator) -> JapaneseNamedEntitiesIdentificator:
#         """Processes the response and attempts country identification."""
#         countries_identified = await cls.llm_builder(response.model_dump())  # Call llm_builder with the response
#
#         if countries_identified:
#             country_names = [entity['word'] for entity in countries_identified if
#                              entity['entity'] == 'LOC']  # Filter country entities
#             response.country_ner = country_names  # Populate with identified countries
#         else:
#             # If no countries identified, handle re-check logic
#             retries = 3
#             for _ in range(retries):
#                 countries_identified = await cls.llm_builder(response.model_dump())
#                 if countries_identified:
#                     country_names = [entity['word'] for entity in countries_identified if entity['entity'] == 'LOC']
#                     response.country_ner = country_names
#                     break
#
#             if not countries_identified:
#                 response.country_ner = [None]  # No countries identified after retries
#                 logger.warning("No countries identified after re-check attempts.")
#
#         return response
#
#
# # Example usage
# async def main():
#     input_text = "Sankei_news そんなの提供できるなら、日本に来てるウクライナ難民も帰国ですね"
#     response = JapaneseNamedEntitiesIdentificator()  # Your response model, make sure it's initialized
#     result = await ModelBuilder.country_identifier(response)
#     logger.info(f"Final response: {result.country_ner}")
#
# # Run the example usage
# # asyncio.run(main())  # Uncomment to run this in an async environment
