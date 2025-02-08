# todo:
#  1) delete stop words,
#  2) delete punctuation,
#  3) delete all odd data (consider how to do that since it's japanese)

class PreprocessingFormula:

    @staticmethod
    async def remove_stop_words():
        ...

    @staticmethod
    async def remove_punctuation():
        ...

    @staticmethod
    async def apply_lemma():
        ...

    @classmethod
    async def preprocessing_pipeline(cls, data: list[str]) -> list[str] | None:

        ...
