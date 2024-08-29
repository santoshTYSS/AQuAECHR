""" 
This strategy is inspired by RARR (https://arxiv.org/pdf/2210.08726)

Steps:
1. Have the model decide if evidence is desired for a given statement (we only consider more that 30 letters a statement/sentence)
2. If yes, have the retriever retrieve k evidences for the statement
3. Have the model decide if the evidence agrees, disagrees or is irrelevant to the statement
4. If the evidence agrees, add it to the evidence pool; if it disagrees, add it to the disagreement pool; if it is irrelevant, ignore it
5. 
"""

import json
import os
import pandas as pd
from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.llms.llm import LLM
from src.models.citation import Citation, SentenceWithCitations
from src.retrievers.retriever import Retriever
from src.utils.bccolors import (
    print_info,
    print_info_highlighted,
    print_success,
    print_error,
    print_warning,
)
from src.utils.get_sentences import get_sentences
from Levenshtein import distance
from langchain_core.documents import Document


RETRIEVE_EVIDENCE_PROMPT = """
You will be determining if a sentence of the answer should be supported by a case law citation.

## Context
Question: {question}
Answer: {answer}

## Now we will analyze if the following sentence should have a citation

The sentence for which to decide if it should have a citation: 
<sentence-start>{sentence}</sentence-end>

First, analyze the sentence and decide if it should be supported by a case law citation.
Note:
- General knowledge, headers, and other non-sentences do not require citations.
- Legal arguments, facts, examples, ... should be supported by citations.

The format of your response MUST look like this:
Thoughts: [Reason why the sentence <sentence-start>{sentence}</sentence-end> should have a citation]
Should have a supporting citation: [Yes/No]
"""


def retrieve_evidence(
    llm: LLM,
    question: str,
    answer: str,
    sentence: str,
):
    prompt = RETRIEVE_EVIDENCE_PROMPT.format(
        question=question,
        answer=answer,
        sentence=sentence,
    )
    response = llm.infer_completion(prompt)

    IDENTIFIER = "Should have a supporting citation:"

    if IDENTIFIER not in response:
        # the llm failed to follow the prompt
        print_error("Error: LLM failed to follow the evidence retrieval prompt.")
        return False

    final_classification = response.split(IDENTIFIER)[1].strip()
    if "yes" in final_classification.lower():
        return True

    return False


AGREEMENT_PROMPT = """
You will be determining if a piece of evidence agrees with, disagrees with, or is irrelevant to an sentence for a given question.

## Context
Question: {question}
Answer: {answer}

## Now we will analyze the evidence for the following sentence in the answer

Evidence: 
<evidence-start>{evidence}</evidence-end>

The sentence for which to decide if the evidence agrees, disagrees, or is irrelevant: 
<sentence-start>{sentence}</sentence-end>

Carefully analyze the evidence and explain in a reasoning step whether it agrees, contradicts, or is irrelevant for the sentence in <sentence-start>...</sentence-end> tags.
Then, based on your reasoning, provide your final classification, which MUST be one of the following:
- Agrees
- Disagrees
- Irrelevant

The format of your response MUST look like this:
Thoughts: [Reason weather the evidence agrees, disagrees or is irrelevant for the given sentence]
Final classification: [Your final classification here: Agrees/Disagrees/Irrelevant]
"""

AGREEMENT = "AGREEMENT"
DISAGREEMENT = "DISAGREEMENT"
IRRELEVANT = "IRRELEVANT"


def agreement(
    llm: LLM,
    question: str,
    answer: str,
    sentence: str,
    evidence: str,
):
    prompt = AGREEMENT_PROMPT.format(
        question=question,
        answer=answer,
        sentence=sentence,
        evidence=evidence,
    )

    response = llm.infer_completion(prompt)

    if "Final classification:" not in response:
        print_error("Error: LLM failed to follow the agreement classification prompt")
        return IRRELEVANT

    final_classification = response.split("Final classification:")[1].strip()

    print_info(f"Classification: {final_classification}")

    if "disagrees" in final_classification.lower():
        return DISAGREEMENT
    if "agrees" in final_classification.lower():
        return AGREEMENT
    if "irrelevant" in final_classification.lower():
        return IRRELEVANT

    # the llm failed to follow the prompt
    print_error("Error: LLM failed to follow the agreement classification prompt")
    return IRRELEVANT


EDIT_PROMPT_DISAGREEMENT = """
You will be editing a sentence based on the disagreement with the evidence.

## Context
Question: {question}
Answer: {answer}

## Now we will edit the following sentence in the answer based on the disagreement with the evidence

Evidence: 
<evidence-start>{evidence}</evidence-end>

The sentence in the answer the evidence disagrees with:
<sentence-start>{sentence}</sentence-end>

First, carefully analyze the sentence and identify the part that contains the disagreement with the evidence.
Then, rewrite the sentence with MINIMAL modification to resolve the disagreement.
We will not accept drastic changes to the sentence!

The format of your response MUST look like this:
Thoughts: [Reason why the sentence in <sentence-start>...</sentence-end> should be edited]
Fix with minimal edit: [The corrected entire sentence with MINIMAL modification to resolve the disagreement enclosed by <fixed-sentence-start>...<fixed-sentence-end>]
"""


def edit_sentence(
    llm: LLM,
    question: str,
    answer: str,
    sentence: str,
    evidence: str,
):
    prompt = EDIT_PROMPT_DISAGREEMENT.format(
        question=question,
        answer=answer,
        sentence=sentence,
        evidence=evidence,
    )

    response = llm.infer_completion(prompt)

    print_info(f"Edit Response:\n{response}\n---\n")

    if "Fix with minimal edit:" not in response:
        return None

    fixed_sentence = response.split("Fix with minimal edit:")[1].strip()

    sentence_start = "<fixed-sentence-start>"
    sentence_end = "</fixed-sentence-end>"
    if sentence_start not in fixed_sentence or sentence_end not in fixed_sentence:
        print_error("Error: LLM failed to follow the edit prompt")
        return None

    fixed_sentence = (
        fixed_sentence.split(sentence_start)[1].split(sentence_end)[0].strip()
    )

    print_info(f"Fixed sentence:\n{response}\n\n")
    print_info(f"Original sentence:\n{sentence}\n\n")

    # I am increasing this as manual observation showed that it prevents the model from making meaningful edits
    # Scenarios include: adding edge cases, adding missing context, etc...
    if distance(sentence, fixed_sentence) > 150:
        # we reject edits with edit distance above 150 characters
        print_error("Error: The edit is too drastic.")
        return None

    print_success("Successfully edited the sentence.")
    return fixed_sentence


TO_BASE_MAP = {
    Experiment.RARR_LLAMA_8B: Experiment.BASE_LLAMA_8B,
    Experiment.RARR_MISTRAL_7B: Experiment.BASE_MISTRAL_7B,
    Experiment.RARR_SAUL_7B: Experiment.BASE_SAUL_7B,
    Experiment.RARR_LLAMA_70B: Experiment.BASE_LLAMA_70B,
}


def rarr_loop(
    llm: LLM,
    retriever: Retriever,
    experiment: Experiment,
    k: int = 3,
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
        question = row[Column.QUESTION]
        sentences = get_sentences(ga)

        sentences_with_citations = []
        total_citations = 0
        for sentence in sentences:
            print()
            print_info("\n ------- New Sentence ------- \n")
            print()
            print_info("Sentence: " + sentence.replace("\n", " ")[:100] + " ...")

            if len(sentence) < 50:
                # this is very likely not a full sentence
                print_info(f"Skipping header or non-sentence")
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=sentence,
                        citations=[],
                    ).model_dump()
                )
                continue

            if not retrieve_evidence(llm, question, ga, sentence):
                print_info(f"Does not need citations.")
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=sentence,
                        citations=[],
                    ).model_dump()
                )
                continue

            docs = retriever.retrieve(sentence, k=k)
            docs = [doc[0] for doc in docs]

            agreement_docs: list[Document] = []
            disagreement_docs: list[Document] = []

            for j, doc in enumerate(docs):
                agreement_classification = agreement(
                    llm, question, ga, sentence, doc.page_content
                )

                if agreement_classification == AGREEMENT:
                    print_success(f"Doc {j}: Agreement found")
                    agreement_docs.append(doc)
                elif agreement_classification == DISAGREEMENT:
                    print_warning(f"Doc {j}: Disagreement found")
                    disagreement_docs.append(doc)
                else:
                    print_info(f"Doc {j}: Irrelevant evidence")
                    continue

            if len(agreement_docs) > 0:
                print_success(f"Adding {len(agreement_docs)} citations.")
                total_citations += len(agreement_docs)
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
                            for doc in agreement_docs
                        ],
                    ).model_dump()
                )
            elif len(disagreement_docs) > 0:
                # we need to edit the sentence
                print_warning(f"sentence with {len(disagreement_docs)} disagreements.")
                fixed_sentence = edit_sentence(
                    llm,
                    question,
                    ga,
                    sentence,
                    "\n".join([doc.page_content for doc in disagreement_docs]),
                )
                if fixed_sentence is None:
                    # we could not fix the sentence
                    print_error("Failed to fix the sentence.")
                    sentences_with_citations.append(
                        SentenceWithCitations(
                            sentence=sentence,
                            citations=[],
                        ).model_dump()
                    )
                    continue

                total_citations += len(disagreement_docs)

                print_success(f"Fixed sentence: {fixed_sentence}")
                print_info(f"Before: {sentence}")
                print_success(f"Adding {len(disagreement_docs)} citations.")
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=fixed_sentence,
                        citations=[
                            Citation(
                                case_id=doc.metadata["case_id"],
                                case_name=doc.metadata["case_name"],
                                paragraph_number=doc.metadata["paragraph_number"],
                                paragraph_text=doc.page_content,
                            )
                            for doc in disagreement_docs
                        ],
                    ).model_dump()
                )
                continue

            else:
                # we could not find any relevant evidence
                print_info(f"No relevant evidence found.")
                sentences_with_citations.append(
                    SentenceWithCitations(
                        sentence=sentence,
                        citations=[],
                    ).model_dump()
                )

        print_info_highlighted(
            f"Row {i}: {total_citations} citations added to the answer.\n\n"
        )
        df.at[i, Column.GENERATED_CITATIONS] = json.dumps(
            sentences_with_citations, indent=4
        )
        df.to_csv(path, index=False)
