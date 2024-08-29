import re
from pydantic import BaseModel


def remove_paragraph_number(text: str):
    text = text.strip()
    text = re.sub(r"^\d*\.", "", text)
    text = text.strip()
    return text


class TargetCitation(BaseModel):
    case_name: str
    case_id: str
    paragraph_numbers: list[int]
    paragraphs_map: dict[str, str]

    def citation_block(self):
        self.paragraphs_map = {
            paragraph_number: remove_paragraph_number(paragraph_text)
            for paragraph_number, paragraph_text in self.paragraphs_map.items()
        }
        return " ".join(self.paragraphs_map.values())
