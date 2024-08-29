import time
import pandas as pd
from langchain_core.documents import Document

from src.retrievers.gtr_t5 import get_db

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


db = get_db()

entires = db._collection.count()

print("Entries:", entires)

all_paragraphs = all_paragraphs[entires + 1 :]

# Process paragraphs in batches to avoid memory issues
n = 2000
while all_paragraphs:
    paragraphs = all_paragraphs[:n]
    all_paragraphs = all_paragraphs[n:]

    # Start timing
    start_time = time.time()

    # Add documents to Chroma
    db.add_documents(documents=paragraphs)

    # Stop timing
    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"Processed {len(paragraphs)} paragraphs ({int(minutes)}:{int(seconds)})")
    print(f"Remaining {len(all_paragraphs)} paragraphs")
