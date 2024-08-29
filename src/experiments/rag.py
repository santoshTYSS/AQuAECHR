import json
import pandas as pd
from langchain_core.documents import Document

from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.llms.llm import LLM
from src.models.citation import Citation, SentenceWithCitations
from src.retrievers.gtr_t5 import GTR
from src.retrievers.retriever import Retriever
from src.utils.text_utils import get_sentences, get_citations, remove_citations


def generate_response(question: str, llm: LLM, retriever: Retriever, k: int = 10):
    docs = retriever.retrieve(question, k=k)

    docs = [doc[0] for doc in docs]

    paragraphs = "\n".join(
        [
            f"[Doc {i+1}]: "
            + GTR.remove_par_text_prefix(
                doc.page_content,
                doc.metadata["case_name"],
                doc.metadata["paragraph_number"],
            )
            for i, doc in enumerate(docs)
        ]
    )

    prompt = f"""
    You are an ECHR legal expert tasked to answer a question.
    The following documents were retrieved and should help you answer the question:
    {paragraphs}

    Instructions:
    Use the retrieved documents to answer the question. 
    Reuse the language from the documents!
    Cite relevant documents at the end of a sentence! 
    Accepted formats: sentence [citation(s)].
    Valid citation formats: [Doc 1] or [Doc 1, Doc 2, Doc 3]
    You must follow the [Doc i] format! Do NOT use the case names or paragraph numbers to cite documents!
    You should NOT provide a list of all used citations at the end of your response!

    Question: {question}
    Answer:
    """

    response = llm.infer_completion(prompt)
    return response, docs


def parse_response(response: str, docs: list[Document]):
    response_sentences = get_sentences(response)
    citations = []
    for sentence in response_sentences:
        doc_numbers = get_citations(sentence)
        doc_numbers = [d - 1 for d in doc_numbers if d - 1 < len(docs) and d - 1 >= 0]
        docs_in_sentence = [docs[doc_number] for doc_number in doc_numbers]

        citations.append(
            SentenceWithCitations(
                sentence=remove_citations(sentence),
                citations=[
                    Citation(
                        case_id=doc.metadata["case_id"],
                        case_name=doc.metadata["case_name"],
                        paragraph_number=doc.metadata["paragraph_number"],
                        paragraph_text=doc.page_content,
                    )
                    for doc in docs_in_sentence
                ],
            ).model_dump()
        )

    parsed_response = "\n".join([remove_citations(s) for s in response_sentences])

    return parsed_response, citations


def rag_loop(llm: LLM, retriever: Retriever, experiment: Experiment, k: int = 10):
    df, path = load_experiment_df(experiment)

    for i, row in df.iterrows():
        if Column.GENERATED_ANSWER in row and pd.notnull(row[Column.GENERATED_ANSWER]):
            print("Skipping row", i, "as it already has a response")
            continue
        print("Processing row", i)
        question = row["question"]
        response, docs = generate_response(question, llm, retriever, k)
        print(f"Question:\n{question}")
        print(f"Response:\n{response}")
        print()
        parsed_response, citations = parse_response(response, docs)
        df.at[i, Column.GENERATED_ANSWER] = parsed_response
        citations = json.dumps(citations, indent=4)
        df.at[i, Column.GENERATED_CITATIONS] = citations
        print()
        print(f"Parsed Response:\n{parsed_response}")
        print()
        print(f"Citations:\n{citations}")
        print()
        print(" --- ")
        print()
        df.to_csv(path, index=False)
