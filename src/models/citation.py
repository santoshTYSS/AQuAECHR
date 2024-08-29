from pydantic import BaseModel


class Citation(BaseModel):
    case_id: str
    case_name: str
    paragraph_number: int
    paragraph_text: str

    def __eq__(self, other):
        if isinstance(other, Citation):
            return (
                self.case_id == other.case_id
                and self.paragraph_number == other.paragraph_number
            )
        return False

    def __hash__(self):
        return hash((self.case_id, self.paragraph_number))


class SentenceWithCitations(BaseModel):
    sentence: str
    citations: list[Citation]
