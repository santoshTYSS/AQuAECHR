from pydantic import BaseModel


class ClaimRecall(BaseModel):
    ga_entails_ta_i: list[float]
    claim_recall: float
