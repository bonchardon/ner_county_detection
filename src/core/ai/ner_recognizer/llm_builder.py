from os import environ
from re import search, findall, DOTALL
from typing import List, Union

from loguru import logger

from dotenv import load_dotenv

import torch
from transformers import (
    pipeline,
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TokenClassificationPipeline,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    GenerationConfig,
    Pipeline,
)

from huggingface_hub import login

from langchain_core.prompts import PromptTemplate

# from core.ai.ner_recognizer.enums import AiModels
from core.ai.models import JapaneseNamedEntitiesIdentificator, IndirectMentioning
from core.train_test_set.corpus import DataSet

load_dotenv()
login(token=environ.get('LLAMA_TOKEN'))


class ModelBuilder:
    @staticmethod
    async def llm_builder():
        model = AutoModelForCausalLM.from_pretrained(
            'elyza/ELYZA-japanese-Llama-2-7b-instruct',
            torch_dtype=torch.float16,
            trust_remote_code=True, 
            device_map='auto',
        )
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            'elyza/ELYZA-japanese-Llama-2-7b-instruct', 
            use_fast=True,
        )
        generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name='elyza/ELYZA-japanese-Llama-2-7b-instruct',
            temperature=0.2,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            max_new_tokens=150,
            repetition_penalty=1.2,
            num_beams=5,
        )
        if not (text_pipeline := pipeline(
            task='text-generation',
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            generation_config=generation_config,
        )):
            logger.warning('There is an issue when combining text pipeline.')
            return
        return text_pipeline

    @staticmethod
    async def prompt_template(input_text: Union[list[DataSet], list[str], str], prompt: str) -> str:
        if not isinstance(prompt, str):
            logger.error('Prompt is not a string.')
            return
        prompt_template = PromptTemplate.from_template(prompt)
        return prompt_template.format(input_text=input_text)
    
    @staticmethod
    async def extract_ner(response: List) -> List:
        if not (response := response[0]['generated_text']):
            return
        if not (match := search(r'\|begin_of_text\|\s*(.*?)\s*\|end_of_text\|', response, DOTALL)):
            return
        content = match.group(1)
        return findall(r'[\w一-龥ぁ-んァ-ンー]+', content)

    @classmethod
    async def prompt_ner(cls, input_text: Union[list[DataSet]]):
        formatted_input: str = await cls.prompt_template(
            input_text=input_text,
            prompt=JapaneseNamedEntitiesIdentificator().country_ner
        )
        logger.info(formatted_input)
        llm_pipeline: Pipeline | None = await cls.llm_builder()
        if llm_pipeline is None:
            logger.error('NER pipeline could not be loaded.')
            return
        response_japanese: list = llm_pipeline(formatted_input)
        logger.info(response_japanese)
        if response_japanese[0].get('generated_text') == '':
            logger.info('No countries identified.')
            # todo: recheck and apply RAG
            return ('"ner": None')
        return response_japanese






    # @classmethod
    # async def check_for_indirect_mentions(cls, input_text: Union[list[DataSet]]):
    #     """ We are about to use RAG, so we can identify any indirect mentioning. """
    #     formatted_input: str = await cls.prompt_template(
    #         input_text=input_text,
    #         prompt=IndirectMentioning().augmented_generation_check
    #     )
    #     llm_pipeline: Pipeline | None = await cls.llm_builder()
    #     if llm_pipeline is None:
    #         logger.error('NER pipeline could not be loaded.')
    #         return
    #     result: str = llm_pipeline(formatted_input)
    #     logger.info(f'NER Result: {result}')
    #     if result is False:
    #         logger.info('There is no indirect mentioning.')
    #         return (
    #             '<|begin_of_text|> Go ahead and try to analyze the given text 2 more times. '
    #             'In case no indirect named entities are still identified, the "reply" section has to be None; like so: '
    #             '"result": [None]'
    #             '<|end_of_text|>'
    #         )
    #     return result
