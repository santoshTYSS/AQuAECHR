from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


class T5Embeddings:

    def __init__(self, device: str = "cpu") -> None:
        self.model = SentenceTransformer(
            "sentence-transformers/gtr-t5-xl", cache_folder="cache/", device=device
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.tolist()
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.model.encode([text], convert_to_numpy=True)
        return embeddings.tolist()[0]


def get_db():
    db = Chroma(
        persist_directory="./data/chroma_gtr_t5_xl_db",
        embedding_function=T5Embeddings(),
    )
    print("DB Loaded - Entries:", db._collection.count())
    return db


def remove_par_text_prefix(text: str, case_name: str, paragraph_number: str):
    text = text.replace(f"{case_name}; § {paragraph_number}: ", "")
    return text


class GTR:
    def __init__(self):
        self.db = get_db()

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        return self.db.similarity_search_with_score(query, k=k)

    @staticmethod
    def remove_par_text_prefix(text: str, case_name: str, paragraph_number: str | int):
        text = text.replace(f"{case_name}; § {paragraph_number}: ", "")
        return text
