from pydantic import BaseModel


class RougeScore(BaseModel):
    rouge_1: float
    rouge_2: float
    rouge_l: float
    rouge_sum: float
