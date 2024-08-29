""" This code is inspired by LLatrieval (https://arxiv.org/pdf/2311.07838) """

import json
import re
import pandas as pd
from langchain_core.documents import Document

from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.llms.llm import LLM
from src.models.citation import Citation, SentenceWithCitations
from src.retrievers.gtr_t5 import GTR
from src.retrievers.retriever import Retriever
from src.utils.text_utils import get_sentences, get_citations, remove_citations


def get_digit_sequences_as_ints(s):
    # Find all sequences of digits in the string
    digit_sequences = re.findall(r"\d+", s)
    # Convert these sequences to integers
    return [int(num) for num in digit_sequences]


# Step 1: Score and filter
SCORE_PROMPT = """
You are ScoreGPT as introduced below.
You are ScoreGPT, capable of scoring candidate documents based on their level of support for the corresponding question, with a rating range from 0 to 10.

Input:
- Question: The specific question.
- Candidate Documents: Documents whose combination may maximally support the corresponding question.

Skill:
1. Analyzing the given question(s) and understanding the required information.
2. Searching through documents to score them based on their level of support for the corresponding question(s),
with a rating range from 0 to 10.

Output:
- A score ranging from 0 to 10, where a higher score indicates greater support of the candidate documents for the corresponding question, and a lower score indicates lesser support.

Candidate Documents:
{documents}

Question: 
{question}

Output Format: (You MUST follow this output format!)
Thoughts: [Your thoughts about how well the candidate documents support the question]
Score: [SCORE]
"""


def score(q: str, D: list[Document], llm: LLM) -> float:
    """
    Input:
        Question q, document pool D, the large language model LLM
    Output:
        Score S
    """
    prompt = SCORE_PROMPT.format(
        documents="\n".join(
            [f"Doc {i+1}: " + doc.page_content for i, doc in enumerate(D)]
        ),
        question=q,
    )
    response = llm.infer_completion(prompt)
    print(f"Score Response:\n{response}\n")
    if "Score:" not in response:  # failsafe
        print("No score found")
        return 0

    response = response.split("Score:")[1].strip()

    digits = get_digit_sequences_as_ints(response)
    if len(digits) == 0:  # failsafe
        print("No score found")
        return 0
    score = get_digit_sequences_as_ints(response)[0]
    print(f"Verification score: {score}")
    return score


# Step 2: Select Progressively
PROGRESSIVE_SELECTION_PROMPT = """
You are DocSelectorGPT, capable of selecting a specified number of documents for answering the user's specific question.

Input:
- Question: The specific question
- Candidate Documents: Documents contain supporting documents which can support answering the given questions. Candidate documents will have their own identifiers for FactRetrieverGPT to cite.

Skill:
1. Analyzing the given question and understanding the required information.
2. Searching through candidate documents to select k supporting documents whose combination can maximally support giving a direct, accurate, clear and engaging answer to the question and make the answer and is closely related to the core of the question.

Workflow:
1. Read and understand the questions posed by the user.
2. Browse through candidate documents to select k documents whose combination can maximally support giving a direct, accurate, clear and engaging answer to the question(s) and make the answer and is closely related to the core of the question.
3. List all selected documents.

Output:
- Selected Documents: The identifiers of selected supporting documents whose combination can maximally support giving an accurate and engaging answer to the question and make the answer and is closely related to the core of the question.

Output Example:
Selected Documents: Doc 2, Doc 6, Doc 8 (You MUST follow this format!)

Max number of selectable documents: {k}
- You can only select a maximum of {k} documents!

Candidate Documents:
{documents}

Question: 
{question}

Output Format (You MUST follow this output format!)
Thoughts: [Your thoughts about which candidate documents support the question well and why]
Selected Documents: [document identifiers] 
"""


def progressive_selection(
    q: str, D: list[Document], llm: LLM, k: int
) -> list[Document]:
    """
    Input:
        Question q, document pool D, the large language model LLM, the retriever R, the max quantity of selected documents k
    Output:
        Selected documents S
    """
    prompt = PROGRESSIVE_SELECTION_PROMPT.format(
        documents="\n".join(
            [f"Doc {i+1}: " + doc.page_content for i, doc in enumerate(D)]
        ),
        question=q,
        k=k,
    )
    response = llm.infer_completion(prompt)

    print(f"Progressive Selection Response:\n{response}\n")

    if "Selected Documents:" not in response:  # failsafe
        return D

    selected_docs_string = response.split("Selected Documents:")[1].strip()

    print(f"Selected documents: {selected_docs_string}")
    selected_docs = get_digit_sequences_as_ints(selected_docs_string)
    selected_docs = [D[doc - 1] for doc in selected_docs if doc - 1 < len(D)]
    return selected_docs


# Step 3: Passage retrieval
PASSAGE_RETRIEVAL_PROMPT = """
You are a PassageRetriever, 
capable of identifying missing content that answers the given question but does not exist in the given possible answering
passages and then using your own knowledge to generate correct answering passages using missing content you identify.

Input:
- Question: The specific question.
- Answering Passages: Possible answering passages.

Output:
- Correct answering passages generated using missing content you identify based on your own knowledge.

Rules:
1. You have to use your own knowledge to generate correct answering passages using missing content you identify.
2. Only generate the required correct answering passages. Do not output anything else.
3. Directly use your own knowledge to generate correct answering passages if you think the given possible answering passages do not answer to the given question.

Workflow:
1. Read and understand the question and possible answering passages.
2. Identify missing content that answers the given question but does not exist in the given possible answering passages.
3. Directly use your own knowledge to generate correct answering passages if you think the given possible answering passages do not answer to the given question(s). 
4. Use your own knowledge to generate correct answering passages using missing content you identify.

Answering Passages:
{documents}

Question: 
{question}

Output Format: (You MUST follow this output format!)
Correct Answering Passages: [correct answering passages]
Missing Passages: [missing passages]
"""


def passage_retrieval_prompt(q: str, D: list[Document], llm: LLM, R: Retriever, N: int):
    """
    Input:
        Question q, document pool D, the large language model LLM, the retriever R, the max quantity of missing information N
    Output:
        Missing information M
    """
    prompt = PASSAGE_RETRIEVAL_PROMPT.format(
        documents="\n".join(
            [f"Doc {i+1}: " + doc.page_content for i, doc in enumerate(D)]
        ),
        question=q,
    )
    response = llm.infer_completion(prompt)

    print(f"Passage Retrieval Response:\n{response}\n")

    if "Missing Passages:" not in response:  # failsafe if there are no missing passages
        print("No missing information found")
        return []

    missing_info = response.split("Missing Passages:")[1].strip()

    if len(missing_info.split()) <= 8:  # there are no missing information
        print("Likely no missing information")
        return []

    print(f"Missing information: {missing_info}")
    docs = R.retrieve(missing_info, k=N)
    docs = [doc[0] for doc in docs]

    print(
        f"New Docs ({len(docs)}): {[(doc.metadata['case_id'], doc.metadata['paragraph_number']) for doc in docs]}"
    )
    return docs


def update_docs(D1: list[Document], D2: list[Document]):
    """
    Input:
        Two lists of documents D1 and D2
    Output:
        Updated document list D1
    """

    def is_in(doc: Document, docs: list[Document]):
        for d in docs:
            if (
                d.metadata["case_id"] == doc.metadata["case_id"]
                and d.metadata["paragraph_number"] == doc.metadata["paragraph_number"]
            ):
                return True
        return False

    documents = D1
    for doc in D2:
        if not is_in(doc, documents):
            documents.append(doc)
        else:
            print(
                f"Document {doc.metadata['case_id']} paragraph {doc.metadata['paragraph_number']} is already in the list"
            )

    return documents


def llatrieval(q: str, llm: LLM, R: Retriever, T: int, N: int, τ: int):
    """
    Input:
        Question q, document pool D, the large language model LLM, the retriever R, the maximum iteration T,
        each iteration's document candidates quantity N, Verify-Result = Yes if ScoreD ≥ τ else No
    Output:
        Supporting Documents D
    """
    # we retrieve initial documents
    D = R.retrieve(q, k=N - 2)
    D = [doc[0] for doc in D]

    for i in range(T):
        print(f"Starting iteration {i + 1}")
        print(
            f"Documents ({len(D)}): {[(doc.metadata['case_id'], doc.metadata['paragraph_number']) for doc in D]}"
        )

        if D != [] and len(D) < N:
            D = update_docs(D, passage_retrieval_prompt(q, D, llm, R, N))
            print(
                f'Updated documents ({len(D)}): {[(doc.metadata["case_id"], doc.metadata["paragraph_number"]) for doc in D]}'
            )

        sliding_D = D.copy()
        while len(sliding_D) > N:
            filter_docs = sliding_D[0 : N + 2]
            remaining_docs = sliding_D[N + 2 :]

            selected_docs = progressive_selection(q, filter_docs, llm, N)
            if len(selected_docs) > N:  # failsafe if too many were selected
                print(f"Selected too many documents: {len(selected_docs)}")
                selected_docs = selected_docs[0:N]

            sliding_D = selected_docs + remaining_docs
            print(
                f'Remaining documents ({len(sliding_D)}): {[(doc.metadata["case_id"], doc.metadata["paragraph_number"]) for doc in sliding_D]}'
            )

        D = sliding_D
        if score(q, D, llm) >= τ:
            break

    return D


def generate_response(docs: list[Document], question: str, llm: LLM):
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
    return response


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


def llatrieval_loop(
    llm: LLM,
    retriever: Retriever,
    experiment: Experiment,
    k: int = 10,
    T: int = 3,
    τ: int = 8,
):
    df, path = load_experiment_df(experiment)

    for i, row in df.iterrows():
        if Column.GENERATED_ANSWER in row and pd.notnull(row[Column.GENERATED_ANSWER]):
            print("Skipping row", i, "as it already has a response")
            continue
        print("Processing row", i)
        question = row["question"]
        docs = llatrieval(question, llm, retriever, T=T, N=k, τ=τ)
        response = generate_response(docs, question, llm)
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
