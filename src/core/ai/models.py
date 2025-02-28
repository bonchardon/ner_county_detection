# todo:
#  1) find a model according to the needs (find out more on what do we actually need,
#   what type of data we're about to work with);
#  2) train (supervised/unsupervised/reinforcement leaning (?)) model using train set collected;
#  3) collect confusion matrix and gradient descent results to enhance out model in future.

from typing import Type

from loguru import logger

from pydantic import BaseModel, Field

from langchain.schema import AIMessage
from langchain import PromptTemplate


class IsNERPresent(BaseModel):
    check_if_present: bool = Field(
        description='<|begin_of_text|> <|end_of_text|> '
    )


class JapaneseNamedEntitiesIdentificator(IsNERPresent):
    # todo: apply rag here for contextual references
    country_ner: str | None = Field(
        description=(
            '<|begin_of_text|> '
            'You need to identify any country name in a japanese text. '
            'Mind that there can be contextual references of countries, as well. If so, apply RAG.'
            'If there are multiple countries mentioned make sure, you identify ALL the countries.'
            '### Example 1: '
            'Input: amisweetheart 日本まで、日本での住所も決めずに来れる奴らが難民か？ウクライナ人はホームステイして良いですよとか、'
            'たまたま家族がいたからとかで来た人多いと思うけど。 変なYouTubeに感化されて、好き勝手できそうで来てるんやないの？ '
            'ただで電車に乗る方法とか、暴言吐いても射殺されないとか見てさ。'
            'Reply: ["ウクライナ", "日本"]'
            '### Example 2: '
            'Input: ロシア大使館かウクライナ大使館に亡命申請してみたらどうだろう？ 「日本は人権を尊重する国と思ったのに…」'
            '難民審査待たされ野宿3カ月 行き場をなくした外国人が増えている：東京新聞 TOKYO Web'
            'Reply: ["ロシア", "ウクライナ"]'
            '<|end_of_text|>'
        ),
        default=False
    )


class CountriesReCheck(BaseModel):
    result: str | None = Field(
        description=(
            '<|begin_of_text|> '
            'If there no countries identified, recheck and analyze the text maximum 3 more times. '
            'In case, there are still not named entities identified, give reply as mentioned above:'
            'Reply: "result": [None]'
            '<|end_of_text|>'
        ),
        default=False
    )
