from pydantic import BaseModel


class BertScore(BaseModel):
    precision: float
    recall: float
    f1: float
