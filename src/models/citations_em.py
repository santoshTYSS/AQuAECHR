from pydantic import BaseModel


class CitationsExactMatch(BaseModel):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
