import json
import os
import pandas as pd
from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.models.citation import Citation, SentenceWithCitations
from src.retrievers.retriever import Retriever
from src.utils.get_sentences import get_sentences


TO_BASE_MAP = {
    Experiment.POST_HOC_LLAMA_8B: Experiment.BASE_LLAMA_8B,
    Experiment.POST_HOC_MISTRAL_7B: Experiment.BASE_MISTRAL_7B,
    Experiment.POST_HOC_SAUL_7B: Experiment.BASE_SAUL_7B,
    Experiment.POST_HOC_LLAMA_70B: Experiment.BASE_LLAMA_70B,
}


def post_hoc_loop(
    retriever: Retriever,
    experiment: Experiment,
    threshold: float,
):
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

            docs = retriever.retrieve(sentence, k=1)
            doc = docs[0][0]
            score = docs[0][1]
            if score < threshold:
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=sentence,
                        citations=[],
                    ).model_dump()
                )
            else:
                total_citations += 1
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
                        ],
                    ).model_dump()
                )

        print(f"Row {i}: added {total_citations} citations to the answer.")
        df.at[i, Column.GENERATED_CITATIONS] = json.dumps(
            sentences_with_citations, indent=4
        )
        df.to_csv(path, index=False)
