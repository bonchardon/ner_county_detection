from pydantic import BaseModel


class DataSet(BaseModel):
    sentence: list[str]
