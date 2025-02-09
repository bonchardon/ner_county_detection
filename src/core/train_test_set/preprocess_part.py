# todo:
#  1) delete stop words,
#  2) delete punctuation,
#  3) delete all odd data (consider how to do that since it's japanese)

from loguru import logger

from stopwordsiso import stopwords

import torch
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

from core.train_test_set.consts import JAPANESE_SEPARATOR, JAPANESE_PUNCTUATION
from core.train_test_set.corpus import DataSet


class PreprocessingFormula:
    @staticmethod
    async def data_tokenization(sentences: list[str]) -> list[str] | None:
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(JAPANESE_SEPARATOR)
        if not (tokenized_sentences := [
            tokenizer.decode(tokenizer(sent, return_tensors='pt')['input_ids'][0], skip_special_tokens=True)
            for sent in sentences
        ]):
            logger.error('There is an error when tokenizing data.')
            return
        return tokenized_sentences

    @staticmethod
    async def remove_stop_words(sentences: list[str]) -> list[list[str]] | None:
        stop_words: set[str] = set(stopwords('ja'))
        if not (filtered_sentences := [
            [word for word in sent.split() if word not in stop_words and word not in JAPANESE_PUNCTUATION]
            for sent in sentences
        ]):
            logger.error('Error when filtering sentences from stop words.')
            return
        return filtered_sentences

    @staticmethod
    async def apply_lemma():
        ...

    @classmethod
    async def preprocessing_pipeline(cls, data_set: list[str]) -> list[DataSet] | None:
        tokenized_data: list[str] | None = await cls.data_tokenization(sentences=data_set)
        data_sans_stopwords = await cls.remove_stop_words(sentences=tokenized_data)

        preprocessed_data: list[DataSet] = []
        for sentence in data_sans_stopwords:
            data: DataSet = DataSet(sentence=sentence)
            preprocessed_data.append(data)
        return preprocessed_data
