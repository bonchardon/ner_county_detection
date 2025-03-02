# from typing import Any
#
# import torch
# from loguru import logger
# from torch.distributed.pipeline.sync.pipeline import Pipeline
#
# from transformers import (
#     pipeline,
#     AutoTokenizer,
#     PreTrainedTokenizerFast,
#     AutoModelForCausalLM,
#     GenerationConfig,
# )
#
# from langchain_core.prompts import PromptTemplate
#
# from core.ai.enums import AiModels
# from core.ai.models import JapaneseNamedEntitiesIdentificator, CountriesReCheck
#
#
# class ModelBuilder:
#     async def llm_builder(self) -> Pipeline | None:
#         model = AutoModelForCausalLM.from_pretrained(
#             AiModels.LLM_MODEL_3_1_SWALLOW,
#             torch_dtype=torch.float16,
#             trust_remote_code=True
#         )
#         tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(AiModels.LLM_MODEL_3_1_SWALLOW)
#         if not (generation_config := GenerationConfig.from_pretrained(
#             pretrained_model_name=AiModels.LLM_MODEL_3_1_SWALLOW,
#             temperature=0.8,
#             top_p=0.95,
#             top_k=10,
#             max_new_tokens=1024,
#             tensor_parallel_size=1,
#             trust_remote_code=True
#
#         )):
#             logger.error('Issues when generating config.')
#             return
#         if not (text_pipeline := pipeline(
#             task='ner',
#             model=AiModels.LLM_MODEL_3_1_SWALLOW,
#             tokenizer=tokenizer,
#             return_full_text=True,
#             generation_config=generation_config
#
#         )):
#             logger.warning('There is an issue when combining text pipeline.')
#             return
#         return text_pipeline
#
#     async def prompt_ner(self, input_text: str):
#         prompt_template_str = JapaneseNamedEntitiesIdentificator.model_fields['country_ner'].field_info.description
#         prompt_template = PromptTemplate.from_template(prompt_template_str)
#         llm_pipeline = await self.llm_builder()
#         formatted_input = prompt_template.format(input_text=input_text)
#         result = llm_pipeline(formatted_input)
#         logger.info(result)
#         return result
#

from typing import Any
import torch
from loguru import logger
from transformers import (
    pipeline,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    GenerationConfig,
)

from langchain_core.prompts import PromptTemplate

from core.ai.enums import AiModels
from core.ai.models import JapaneseNamedEntitiesIdentificator, CountriesReCheck, CheckRAG


class ModelBuilder:
    async def llm_builder(self) -> pipeline | None:
        model = AutoModelForCausalLM.from_pretrained(
            AiModels.LLM_MODEL_3_1_SWALLOW,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(AiModels.LLM_MODEL_3_1_SWALLOW)
        generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name=AiModels.LLM_MODEL_3_1_SWALLOW,
            temperature=0.8,
            top_p=0.95,
            top_k=10,
            max_new_tokens=1024,
            tensor_parallel_size=1,
            trust_remote_code=True
        )

        # Initialize the NER pipeline
        ner_pipeline = pipeline(
            task='ner',
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            generation_config=generation_config
        )
        return ner_pipeline

    async def prompt_ner(self, input_text: str):
        prompt_template_str = JapaneseNamedEntitiesIdentificator.model_fields['country_ner'].field_info.description
        prompt_template = PromptTemplate.from_template(prompt_template_str)

        # Format input using the template
        formatted_input = prompt_template.format(input_text=input_text)
        logger.info(f"Formatted input for NER: {formatted_input}")

        # Call the NER pipeline
        llm_pipeline = await self.llm_builder()
        if llm_pipeline is None:
            logger.error('NER pipeline could not be loaded.')
            return None

        # Run the NER pipeline
        result = llm_pipeline(formatted_input)
        logger.info(f"NER Result: {result}")

        # Parse the output to extract country names
        country_names = self.extract_country_names(result)
        logger.info(f"Extracted Country Names: {country_names}")
        return country_names

    def extract_country_names(self, ner_results: Any) -> list:
        country_names = []
        for entity in ner_results:
            if entity['entity_group'] == 'LOC' and entity['word'] not in country_names:
                country_names.append(entity['word'])
        return country_names

    async def check_for_indirect_mentions(self, input_text: str):
        prompt_template_str = CheckRAG.model_fields['augmented_generation_check'].field_info.description
        prompt_template = PromptTemplate.from_template(prompt_template_str)

        formatted_input = prompt_template.format(input_text=input_text)
        logger.info(f"Formatted input for RAG Check: {formatted_input}")
        return "Indirect mentions logic not implemented"

