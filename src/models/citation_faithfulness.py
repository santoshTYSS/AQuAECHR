from pydantic import BaseModel


class SentenceCitationsFaithfulness(BaseModel):
    """
    Citation faithfulness of a sentence

    s: sentence
    c: citation
    cc: concatenated citations

    - c_entails_s: for each citation, does it entail the sentence
    - cc_entails_s: do the concatenated citations entail the sentence
    """

    c_entails_s: dict[str, float]
    cc_without_c_entails_s: dict[str, float] | None = None
    cc_entails_s: float


class CitationFaithfulness(BaseModel):
    sentences_citation_faithfulness: list[SentenceCitationsFaithfulness]
    citation_recall: float
    citation_precision: float
