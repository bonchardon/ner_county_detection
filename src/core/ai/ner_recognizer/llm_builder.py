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

from langchain import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from core.ai.enums import AiModels
from core.ai.models import JapaneseNamedEntitiesIdentificator, CountriesReCheck


class ModelBuilder:
    async def llm_builder(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            AiModels.LLM_MODEL_3_1_SWALLOW,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(AiModels.LLM_MODEL_3_1_SWALLOW)
        if not (generation_config := GenerationConfig.from_pretrained(
            pretrained_model_name=AiModels.LLM_MODEL_3_1_SWALLOW,
            temperature=0.8,
            top_p=0.95,
            top_k=10,
            max_new_tokens=1024,
            tensor_parallel_size=1,
            trust_remote_code=True

        )):
            logger.error('Issues when generating config.')
            return
        if not (text_pipeline := pipeline(
            task='ner',
            model=AiModels.LLM_MODEL_3_1_SWALLOW,
            tokenizer=tokenizer,
            return_full_text=True,
            generation_config=generation_config

        )):
            logger.warning('There is an issue when combining text pipeline.')
            return
        return HuggingFacePipeline(
            pipeline=text_pipeline,
            model_kwargs={'temperature': 0}
        )

    async def prompt_ner(self, input_text: str):
        prompt_template_str = JapaneseNamedEntitiesIdentificator.model_fields['country_ner'].field_info.description
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        llm_pipeline = await self.llm_builder()
        formatted_input = prompt_template.format(input_text=input_text)
        result = llm_pipeline(formatted_input)
        logger.info(result)
        return result
