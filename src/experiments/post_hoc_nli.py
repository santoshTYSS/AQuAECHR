""" Note: we did not explore this further """

import json
import os
import pandas as pd
from src.column import Column
from src.constants import CACHE_DIR
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.models.citation import Citation, SentenceWithCitations
from src.retrievers.retriever import Retriever
from src.true_teacher import TRUETeacher
from src.utils.get_sentences import get_sentences
from langchain_core.documents import Document


TO_BASE_MAP = {
    # Experiment.POST_HOC_NLI_LLAMA_8B: Experiment.BASE_LLAMA_8B,
    # Experiment.POST_HOC_NLI_MISTRAL_7B: Experiment.BASE_MISTRAL_7B,
    # Experiment.POST_HOC_NLI_SAUL_7B: Experiment.BASE_SAUL_7B,
    # Experiment.POST_HOC_NLI_LLAMA_70B: Experiment.BASE_LLAMA_70B,
}


def post_hoc_nli_loop(retriever: Retriever, experiment: Experiment, device: str):
    nli = TRUETeacher(device=device, cache_dir=CACHE_DIR)
    base_experiment = TO_BASE_MAP[experiment]
    path = f"data/e_{experiment.value}.csv"
    if not os.path.exists(path):
        df, _ = load_experiment_df(base_experiment)
    else:
        df, _ = load_experiment_df(experiment)

    for i, row in df.iterrows():
        if Column.GENERATED_CITATIONS in row and not pd.isna(
            row[Column.GENERATED_CITATIONS]
        ):
            print(f"Row {i}: already has citations.")
            continue

        ga = row[Column.GENERATED_ANSWER]
        sentences = get_sentences(ga)

        sentences_with_citations = []
        total_citations = 0
        for sentence in sentences:
            if len(sentence) < 50:  # this is likely a header or not a full sentence
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=sentence,
                        citations=[],
                    ).model_dump()
                )
                continue

            docs = retriever.retrieve(sentence, k=20)
            docs = [doc[0] for doc in docs]

            # we take the nli model and find the up to 3 best above 0.5 scores
            best_docs: list[tuple[Document, float]] = []
            for doc in docs:
                nli_score = nli.get_value(doc.page_content, sentence)
                if nli_score > 0.5:
                    best_docs.append((doc, nli_score))

            if len(best_docs) == 0:
                print("No document found")
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=sentence,
                        citations=[],
                    ).model_dump()
                )
                continue

            best_docs_sorted = sorted(best_docs, key=lambda x: x[1], reverse=True)[:3]
            best_docs_sorted = [doc for doc, _ in best_docs_sorted]

            print(f"Adding {len(best_docs_sorted)} citations")
            sentences_with_citations.append(
                SentenceWithCitations(
                    sentence=sentence,
                    citations=[
                        Citation(
                            case_id=doc.metadata["case_id"],
                            case_name=doc.metadata["case_name"],
                            paragraph_number=doc.metadata["paragraph_number"],
                            paragraph_text=doc.page_content,
                        )
                        for doc in best_docs_sorted
                    ],
                ).model_dump()
            )

        total_citations = sum(
            [len(swc["citations"]) for swc in sentences_with_citations]
        )
        print(f"Row {i}: added {total_citations} citations to the answer.")
        df.at[i, Column.GENERATED_CITATIONS] = json.dumps(
            sentences_with_citations, indent=4
        )
        df.to_csv(path, index=False)
