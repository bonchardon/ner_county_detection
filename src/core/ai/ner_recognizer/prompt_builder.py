from typing import Any

from loguru import logger

from langchain_core.prompts import PromptTemplate

from core.ai.models import JapaneseNamedEntitiesIdentificator, CheckRAG


class PromptsBuilder:

    async def prompt_ner(self, input_text: str):
        prompt_template_str = JapaneseNamedEntitiesIdentificator.model_fields['country_ner'].field_info.description
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        formatted_input = prompt_template.format(input_text=input_text)

        logger.info(f'Formatted input for NER: {formatted_input}')
        llm_pipeline = await self.llm_builder()
        if llm_pipeline is None:
            logger.error('NER pipeline could not be loaded.')
            return None

        result = llm_pipeline(formatted_input)
        logger.info(f'NER Result: {result}')

        country_names = self.extract_country_names(result)
        logger.info(f'Extracted Country Names: {country_names}')
        return country_names

    async def extract_country_names(self, ner_results: Any) -> list:
        country_names = []
        for entity in ner_results:
            if entity['entity_group'] == 'LOC' and entity['word'] not in country_names:
                country_names.append(entity['word'])
        return country_names

    async def check_for_indirect_mentions(self, input_text: str):
        prompt_template_str = CheckRAG.model_fields['augmented_generation_check'].field_info.description
        prompt_template = PromptTemplate.from_template(prompt_template_str)

        formatted_input = prompt_template.format(input_text=input_text)
        logger.info(f'Formatted input for RAG Check: {formatted_input}')
        return 'Indirect mentions logic not implemented'
