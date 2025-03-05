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
    country_ner: str | None = Field(
        description=(
            '<|begin_of_text|> '
            'You need to identify any country name in a text in japanese language. '
            'Mind that there can be contextual references of countries, as well. If so, apply RAG.'
            'If there are multiple countries mentioned make sure, you identify ALL the countries.'
            '### Example 1: '
            'Input: amisweetheart 日本まで、日本での住所も決めずに来れる奴らが難民か？ウクライナ人はホームステイして良いですよとか、'
            'たまたま家族がいたからとかで来た人多いと思うけど。 変なYouTubeに感化されて、好き勝手できそうで来てるんやないの？ '
            'ただで電車に乗る方法とか、暴言吐いても射殺されないとか見てさ。'
            'Reply: ["ウクライナ", "日本"]'
            '{'
                '"title_p": "Sankei_news そんなの提供できるなら、日本に来てるウクライナ難民も帰国ですね",'
                '"title_p_countries": ["日本, ウクライナ"],'
                '"result": ['
                    '"日本",'
                    '"ウクライナ"'
                ']'
            '}'
            '### Example 2: '
            'Input: ロシア大使館かウクライナ大使館に亡命申請してみたらどうだろう？ 「日本は人権を尊重する国と思ったのに…」'
            '難民審査待たされ野宿3カ月 行き場をなくした外国人が増えている：東京新聞 TOKYO Web'
            'Reply: '
            '{'
                '"title_p": "ロシア大使館かウクライナ大使館に亡命申請してみたらどうだろう？ 「日本は人権を尊重する国と思ったのに…」'
                            '難民審査待たされ野宿3カ月 行き場をなくした外国人が増えている：東京新聞 TOKYO Web",'
                '"title_p_countries": ['
                '"ロシア",'
                'ウクライナ"'
                '],'
                '"result": ['
                    '"ロシア",'
                    '"ウクライナ",'
                    '"日本"'
                    ']'
            '}'
            '<|end_of_text|>'
        ),
        default=False
    )
