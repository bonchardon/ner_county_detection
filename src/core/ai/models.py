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
            '<|begin_of_text|> '
            'You need to identify any country name in a text in Japanese language. '
            'Mind that there can be contextual references of countries, as well. If so, apply RAG.'
            'If there are multiple countries mentioned make sure, you identify ALL the countries.'
            '### Example 1: '
            'Input: amisweetheart 日本まで、日本での住所も決めずに来れる奴らが難民か？ウクライナ人はホームステイして良いですよとか、'
            'たまたま家族がいたからとかで来た人多いと思うけど。 変なYouTubeに感化されて、好き勝手できそうで来てるんやないの？ '
            'ただで電車に乗る方法とか、暴言吐いても射殺されないとか見てさ。'
            'Reply: ["ウクライナ", "日本"]'
            '<|end_of_text|>'
        ),
        # description='This is the prompt template used for extracting country NERs from Japanese text.'
        description=False
    )
