from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    ...


class IndirectMentioning(AssistantResponse):
    augmented_generation_check: str = Field(
        description=(
            '<|begin_of_text|> You need to identify if there are any indirect mentioning or not. '
            'An example of indirect mentioning: Japan can my named as "the country of the rising sun". '
            'Japan has not been mentioned here directly, yet from the context and general knowledge '
            'we understand that here we are talking about a specific country.'
            'Here apply RAG and from ident'
            '<|end_of_text|>'
        ),
        default=False
    )
 

class JapaneseNamedEntitiesIdentificator(AssistantResponse):
    country_ner: str = Field(
        default=(
            '[INST]'
            '<<SYS>>'
            'You are an assistant tasked with identifying all country names mentioned in a given Japanese text. '
            'Your response should be a JSON object containing a list of all unique country names mentioned, without repetitions or additional information. '
            'Ensure that only internationally recognized sovereign states are included. '
            'If no countries are identified, return {{"ner": None}}.'
            '<</SYS>>'

            'Output the answer in JSON in the following format: {{"ner": ["country_a", "country_b", etc.]}}'
            'Given text: {input_text}'
            '[/INST]'

        ),
        #description='This is the prompt template used for extracting country NERs from Japanese text.'
        description=False
    )
