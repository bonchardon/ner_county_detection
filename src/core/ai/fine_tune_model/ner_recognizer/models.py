# todo:
#  1) find a model according to the needs (find out more on what do we actually need,
#   what type of data we're about to work with);
#  2) train (supervised/unsupervised/reinforcement leaning (?)) model using train set collected;
#  3) collect confusion matrix and gradient descent results to enhance out model in future.

from typing import Type

from loguru import logger

from pydantic import BaseModel, Field

from langchain.schema import AIMessage


class JapaneseNamedEntitiesIdentificator(BaseModel):
    # todo: apply rag here for contextual references
    result: bool = Field(
        description=(
            '<|begin_of_text|> '
            'You need to identify any named entities in japanese text '
            'and/or corpora related to countries. '
            'Or in other words: you have to find names of countries from text data.'
            'Mind that there can be contextual references of countries, as well. If so, apply RAG.'
            'As a reply, you have to give a reply that consists of only NER or country/countries. '
            'If there are multiple countries mentioned make sure, you gie a reply of all the countries mentioned.'
            '### Example 1: '
            'Input: amisweetheart 日本まで、日本での住所も決めずに来れる奴らが難民か？ウクライナ人はホームステイして良いですよとか、'
            'たまたま家族がいたからとかで来た人多いと思うけど。 変なYouTubeに感化されて、好き勝手できそうで来てるんやないの？ '
            'ただで電車に乗る方法とか、暴言吐いても射殺されないとか見てさ。'
            'Reply: ["日本", "ウクライナ"]'
            '### Example 2: '
            'Input: ロシア大使館かウクライナ大使館に亡命申請してみたらどうだろう？ 「日本は人権を尊重する国と思ったのに…」'
            '難民審査待たされ野宿3カ月 行き場をなくした外国人が増えている：東京新聞 TOKYO Web'
            'Reply: ["ロシア", "ウクライナ"]'
            'Make sure you follow the instruction strictly and at the end I receive a short and concise reply, such as:'
            '<|end_of_text|>'
        ),
        default=False
    )


class CountriesReCheck(BaseModel):
    result: bool = Field(
        description=(
            '<|begin_of_text|> '
            'If there no countries identified, recheck and analyze the text maximum 3 more times. '
            'In case, there are still not named entities identified, give reply as mentioned above:'
            'Reply: "result": [None]'
            '<|end_of_text|>'
        ),
        default=False
    )
