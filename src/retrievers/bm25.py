import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

df = pd.read_csv("data/echr_case_paragraphs.csv")

all_paragraphs = [
    Document(
        page_content=f'{case_name}; § {paragraph_number}: {str(paragraph_text).replace(f"{paragraph_number}.", "").strip()}',
        metadata={
            "case_name": case_name,
            "paragraph_number": paragraph_number,
            "case_id": case_id,
        },
    )
    for case_name, paragraph_number, paragraph_text, case_id in zip(
        df["case_name"], df["paragraph_number"], df["paragraph_text"], df["case_id"]
    )
]


class BM25:
    def __init__(self):
        self.retriever = BM25Retriever.from_documents(all_paragraphs)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        docs = self.retriever.invoke(query, k=k)
        docs = [(doc, 0) for doc in docs]
        return docs
