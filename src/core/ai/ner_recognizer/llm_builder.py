from loguru import logger

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    GenerationConfig,
    Pipeline
)

from langchain_core.prompts import PromptTemplate

from core.ai.enums import AiModels
from core.ai.models import JapaneseNamedEntitiesIdentificator, IndirectMentioning


class ModelBuilder:
    @staticmethod
    async def llm_builder() -> Pipeline | None:
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
        if not (text_pipeline := pipeline(
            task='ner',
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            generation_config=generation_config

        )):
            logger.warning('There is an issue when combining text pipeline.')
            return
        return text_pipeline

    @staticmethod
    async def prompt_template(input_text: str, prompt: str) -> str | None:
        if not (prompt_template := PromptTemplate.from_template(prompt)):
            logger.error('There is an issue when trying to use prompt template.')
            return
        return prompt_template.format(input_text=input_text)

    @classmethod
    async def prompt_ner(cls, input_text: str) -> list | str | None:
        formatted_input: str = await cls.prompt_template(
            input_text=input_text,
            prompt=JapaneseNamedEntitiesIdentificator().country_ner
        )

        llm_pipeline: Pipeline | None = await cls.llm_builder()
        if llm_pipeline is None:
            logger.error('NER pipeline could not be loaded.')
            return
        result: list = llm_pipeline(formatted_input)
        logger.info(f'NER Result: {result}')
        if not result:
            logger.info('No countries identified.')
            return (
                '<|begin_of_text|> '
                'If there are no countries identified, recheck and analyze the text one more time. '
                'In case no named entities are still identified, the "reply" section has to be None; like so: '
                '"result": [None]'
                '<|end_of_text|>'
            )
        return result

    @classmethod
    async def check_for_indirect_mentions(cls, input_text: str):
        """ We are about to use RAG, so we can identify any indirect mentioning. """
        formatted_input: str = await cls.prompt_template(
            input_text=input_text,
            prompt=IndirectMentioning().augmented_generation_check
        )
        llm_pipeline: Pipeline | None = await cls.llm_builder()
        if llm_pipeline is None:
            logger.error('NER pipeline could not be loaded.')
            return
        result: str = llm_pipeline(formatted_input)
        logger.info(f'NER Result: {result}')
        if result is False:
            logger.info('There is no indirect mentioning.')
            return (
                '<|begin_of_text|> Go ahead and try to analyze the given text 2 more times. '
                'In case no indirect named entities are still identified, the "reply" section has to be None; like so: '
                '"result": [None]'
                '<|end_of_text|>'
            )
        return result
