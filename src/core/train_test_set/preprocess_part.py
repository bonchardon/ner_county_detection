from re import sub
from typing import Union

from loguru import logger

from stopwordsiso import stopwords

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from core.train_test_set.corpus import DataSet 
from core.train_test_set.consts import JAPANESE_SEPARATOR, JAPANESE_PUNCTUATION, REGEX_NUMBERS


class PreprocessingFormula:
    @staticmethod
    async def data_tokenization(sentences: list[str]) -> Union[list[str], None]:
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(JAPANESE_SEPARATOR)
        if not (tokenized_sentences := [
            tokenizer.decode(tokenizer(sent, return_tensors='pt')['input_ids'][0], skip_special_tokens=True)
            for sent in sentences
        ]):
            logger.error('There is an error when tokenizing data.')
            return
        return tokenized_sentences

    @staticmethod
    async def remove_stop_words(sentences: list[str]) -> Union[list[list[str]], None]:
        stop_words: set[str] = set(stopwords('ja'))
        if not (filtered_sentences := [
            [
                word
                for word in sent.split()
                if word not in stop_words
                and word not in JAPANESE_PUNCTUATION
                and sub(REGEX_NUMBERS, '', word)
            ]
            for sent in sentences
        ]):
            logger.error('Error when filtering sentences from stop words.')
            return
        return filtered_sentences

    @classmethod
    async def preprocessing_pipeline(cls, data_set: list[str]) -> Union[list[DataSet] , None]:
        tokenized_data: list[str] | None = await cls.data_tokenization(sentences=data_set)
        data_sans_stopwords = await cls.remove_stop_words(sentences=tokenized_data)

        preprocessed_data: list[DataSet] = []
        for sentence in data_sans_stopwords:
            data: DataSet = DataSet(sentence=sentence)
            preprocessed_data.append(data)
        return preprocessed_data
