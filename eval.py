"""
Evaluation metrics overview:

Answer Fluency: 
- MAUVE (see mauve_score.py)

Answer Correctness: 
- ROUGE-L (F1 score)
- BERTScore (computed with t5-large)
- Claim recall (TrueTeacher)

Citation Faithfulness:
- Citation Recall & Precision (TrueTeacher)

Evidence Retrieval: 
- EM (precision, recall, and F1)
- Citation Entailment (NLI): are citations in the generated response entailed in the target citation blocks
"""

import json
import os
import sys
import spacy
import argparse
import pandas as pd
import bert_score as bs
from evaluate import load

from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.models import (
    TargetCitation,
    BertScore,
    CitationsExactMatch,
    RougeScore,
    ClaimRecall,
    CitationFaithfulness,
    SentenceCitationsFaithfulness,
    Citation,
    SentenceWithCitations,
)
from src.constants import CACHE_DIR
from src.retrievers.gtr_t5 import GTR
from src.true_teacher import TRUETeacher

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

nlp = spacy.load("en_core_web_trf")
rouge = load("rouge", cache_dir=CACHE_DIR)

device = None
nli = None


def load_nli():
    global nli
    if nli is None:
        nli = TRUETeacher(device=device, cache_dir=CACHE_DIR)


def unload_nli():
    global nli
    nli = None


def get_sentences(text: str):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


def bert_score(generated_text: str, target_text: str):
    P, R, F1 = bs.score(
        cands=[generated_text],
        refs=[target_text],
        lang="en",
        model_type="t5-large",
        # model_type="bert-large-uncased",
        verbose=True,
        device=device,
    )

    return BertScore(
        precision=float(P.item()), recall=float(R.item()), f1=float(F1.item())
    )


def rouge_score(generated_text: str, target_text: str):
    results = rouge.compute(
        predictions=[generated_text], references=[target_text], use_stemmer=True
    )
    return RougeScore(
        rouge_1=results["rouge1"],
        rouge_2=results["rouge2"],
        rouge_l=results["rougeL"],
        rouge_sum=results["rougeLsum"],
    )


def claim_recall(ga: str, ta: str):
    """
    For each statement (sentence) in the target answer we check if it is entailed (0 or 1) in the generated answer.
    Claim recall of a QA pair is the average of these entailments.
    """
    tas = get_sentences(ta)
    ga_entails_ta_i = [nli.get_value(ga, t_i) for t_i in tas]
    claim_recall = sum([1 for e in ga_entails_ta_i if e > 0.5]) / len(tas)
    return ClaimRecall(ga_entails_ta_i=ga_entails_ta_i, claim_recall=claim_recall)


def sentence_citations_faithfulness(generated_sentence: SentenceWithCitations):
    """
    This method calculates the citation faithfulness of a sentence
    """

    cs = generated_sentence.citations
    s = generated_sentence.sentence

    if len(cs) == 1:
        c = cs[0]
        c_entails_s = {c.case_id: nli.get_value(c.paragraph_text, s)}
        cc_entails_s = c_entails_s[c.case_id]

        return SentenceCitationsFaithfulness(
            c_entails_s=c_entails_s,
            cc_entails_s=cc_entails_s,
        )

    c_entails_s = {}
    cc_without_c_entails_s = {}

    cc = " ".join([c.paragraph_text for c in cs])

    for c in cs:
        c_entails_s[c.case_id] = nli.get_value(c.paragraph_text, s)
        cc_without_c = " ".join(
            [ci.paragraph_text for ci in cs if ci.case_id != c.case_id]
        )
        cc_without_c_entails_s[c.case_id] = nli.get_value(
            cc_without_c,
            s,
        )

    cc_entails_s = nli.get_value(cc, s)

    return SentenceCitationsFaithfulness(
        c_entails_s=c_entails_s,
        cc_without_c_entails_s=cc_without_c_entails_s,
        cc_entails_s=cc_entails_s,
    )


def citation_faithfulness(generated_answer: list[SentenceWithCitations]):
    """
    Citation faithfulness of an answer (recall, precision) as defined in "Enabling Large Language Models to Generate Text with Citations"

    citation_recall:
        - is the output entirely supported by the cited passages (1 = yes, 0 = not at all)
        - average of cc_entails_s
    citation_precision: are there irrelevant citations (1 = no, 0 = yes)
        - c is irrelevant if c does not support s AND cc\c does support s
        - precision(c, C, s) = 1 if c is relevant AND recall(s, C) is 1
        - average over all precision(c, C, s)
    """

    def to_binary(value: float):
        return 1 if value > 0.5 else 0

    sentences_citation_faithfulness: list[SentenceCitationsFaithfulness] = []

    for sentence in generated_answer:
        if sentence.citations == []:  # skip sentences without citations
            continue
        sentences_citation_faithfulness.append(
            sentence_citations_faithfulness(sentence)
        )

    if len(sentences_citation_faithfulness) == 0:
        # The answer did not have citations
        # We assume all statements are general statements => we do not consider this answer in our average across all answers
        return CitationFaithfulness(
            sentences_citation_faithfulness=[],
            citation_recall=0,
            citation_precision=0,
        )

    # citation_recall
    citation_statement_recalls = [
        to_binary(scf.cc_entails_s) for scf in sentences_citation_faithfulness
    ]
    answer_citation_recall = sum(citation_statement_recalls) / len(
        citation_statement_recalls
    )

    # citation_precision
    citation_precisions = []
    for res in sentences_citation_faithfulness:
        if not res.cc_without_c_entails_s:
            # we only have one citation
            citation_precisions.append(to_binary(res.cc_entails_s))
            continue

        for c in res.c_entails_s.keys():
            c_is_irrelevant = (
                to_binary(res.c_entails_s[c]) == 0
                and to_binary(res.cc_without_c_entails_s[c]) == 1
            )
            statement_recall = to_binary(res.cc_entails_s)
            if not c_is_irrelevant and statement_recall == 1:
                citation_precisions.append(1)
            else:
                citation_precisions.append(0)

    answer_citation_precision = sum(citation_precisions) / len(citation_precisions)

    return CitationFaithfulness(
        sentences_citation_faithfulness=sentences_citation_faithfulness,
        citation_recall=answer_citation_recall,
        citation_precision=answer_citation_precision,
    )


def citation_similarity_em(gcs: set[Citation], tcs: set[Citation]):
    tp = len(gcs.intersection(tcs))
    fp = len(gcs.difference(tcs))
    fn = len(tcs.difference(gcs))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return CitationsExactMatch(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def citation_similarity_nli(gcs: set[Citation], tcs: list[TargetCitation]):
    """
    For each citation block we check if we have retrieved (and used) a
    citation that is entailed in this block.
    """
    all_scores = []

    if len(gcs) == 0:
        # we should skip responses without citations
        return -1

    for tc in tcs:
        tcb_i = tc.citation_block()
        exists_gc_i_entailed_in_tcb_i = max(
            [
                nli.get_value(premise=tcb_i, hypothesis=gc_i.paragraph_text)
                for gc_i in gcs
            ]
        )
        exists_gc_i_entailed_in_tcb_i = 1 if exists_gc_i_entailed_in_tcb_i > 0.5 else 0
        all_scores.append(exists_gc_i_entailed_in_tcb_i)

    return sum(all_scores) / len(tcs)


def get_generated_sentences_with_citations(row):
    gc = row[Column.GENERATED_CITATIONS]
    gc = json.loads(gc)
    return [SentenceWithCitations.model_validate(g) for g in gc]


def get_generated_citations(row):
    gs = get_generated_sentences_with_citations(row)
    gcs = set([c for g in gs for c in g.citations])
    # we clean the gcs paragraph texts
    for gc in gcs:
        gc.paragraph_text = GTR.remove_par_text_prefix(
            gc.paragraph_text, gc.case_name, gc.paragraph_number
        )
    return gcs


def get_target_citations(row):
    tc = row[Column.TARGET_CITATIONS]
    tc = json.loads(tc)
    target_citations = [TargetCitation.model_validate(t) for t in tc]
    return target_citations


def get_generated_answer(row) -> str:
    if pd.isna(row[Column.GENERATED_ANSWER]):
        # in very rare occasions the generated answer is empty
        # i.e. the model is producing only a citation which we remove
        return "none"
    return row[Column.GENERATED_ANSWER]


def correctness_bert_row(row):
    ga = get_generated_answer(row)
    ta = row[Column.TARGET_ANSWER]
    score = bert_score(ga, ta)
    return score.model_dump_json(indent=4)


def correctness_rouge_row(row):
    ga = get_generated_answer(row)
    ta = row[Column.TARGET_ANSWER]
    score = rouge_score(ga, ta)
    return score.model_dump_json(indent=4)


def claim_recall_row(row):
    ga = get_generated_answer(row)
    ta = row[Column.TARGET_ANSWER]
    score = claim_recall(ga, ta)
    return score.model_dump_json(indent=3)


def citation_faithfulness_row(row):
    ga_wc = get_generated_sentences_with_citations(row)
    score = citation_faithfulness(ga_wc)
    return score.model_dump_json(indent=4)


def citation_similarity_nli_row(row):
    gc = get_generated_citations(row)
    tc = get_target_citations(row)
    score = citation_similarity_nli(gc, tc)
    return score


def citation_similarity_em_row(row):
    def to_general_citations_format(tc: TargetCitation):
        return [
            Citation(
                case_name=tc.case_name,
                case_id=tc.case_id,
                paragraph_number=pn,
                paragraph_text=tc.paragraphs_map[str(pn)],
            )
            for pn in tc.paragraph_numbers
        ]

    tcs = get_target_citations(row)
    tcs = [to_general_citations_format(t) for t in tcs]
    tcs = set([c for tc in tcs for c in tc])  # flatten the list
    gcs = get_generated_citations(row)
    score = citation_similarity_em(gcs, tcs)
    return score.model_dump_json(indent=4)


# each function should return a json dump or float (something to directly save in the data frame)
METRIC_FUNCTIONS = {
    # correctness metrics
    Column.CORRECTNESS_BERT: correctness_bert_row,
    Column.CORRECTNESS_ROUGE: correctness_rouge_row,
    Column.CLAIM_RECALL: claim_recall_row,
    # citation faithfulness
    Column.CITATION_FAITHFULNESS: citation_faithfulness_row,
    # citation similarity
    Column.CITATION_SIMILARITY_EM: citation_similarity_em_row,
    Column.CITATION_SIMILARITY_NLI: citation_similarity_nli_row,
}

NLI_BASED_METRICS = [
    Column.CLAIM_RECALL,
    Column.CITATION_FAITHFULNESS,
    Column.CITATION_SIMILARITY_NLI,
]


def main(args: list[str]):
    parser = argparse.ArgumentParser(description="Generation Evaluation")

    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment to run",
        choices=[e.value for e in Experiment],
        required=True,
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Device to use for evaluation (e.g., 'cpu', 'cuda:0').",
    )

    parser.add_argument(
        "--metrics",
        help="The metrics to evaluate. (e.g., 'correctness_bert, correctness_rouge, claim_recall')",
    )

    parser.add_argument(
        "--overwrite",
        help="Overwrite existing entries.",
    )

    parser.add_argument(
        "--n",
        help="For a lower number of rows to process",
    )

    args = parser.parse_args()

    global device
    device = args.device

    df, path = load_experiment_df(args.experiment)

    if args.metrics:
        metrics = args.metrics.split(", ")
    elif args.metrics == "post":
        print("Post-hoc metrics")
        metrics = [
            Column.CITATION_FAITHFULNESS,
            Column.CITATION_SIMILARITY_NLI,
            Column.CITATION_SIMILARITY_EM,
        ]
    else:
        if "base" in args.experiment:
            metrics = [
                Column.CORRECTNESS_BERT,
                Column.CORRECTNESS_ROUGE,
                Column.CLAIM_RECALL,
            ]
        else:
            metrics = METRIC_FUNCTIONS.keys()

    for metric in metrics:
        for i, row in df.iterrows():
            # if the entry already exists, skip it
            if args.n and i >= int(args.n):
                continue

            if (
                metric in row
                and not pd.isna(row[metric])
                and not args.overwrite == "true"
            ):
                print()
                print(f"Skipping row {i} for metric {metric}: entry already exists")
                continue

            if metric in NLI_BASED_METRICS:
                load_nli()

            print()
            print(f"Calculating metric {metric} for {args.experiment} row {i}")
            result = METRIC_FUNCTIONS[metric](row)
            print(f"Result:\n{result}\n")

            df.at[i, metric] = result

            # we save the file after each row to avoid loosing progress
            df.to_csv(path, index=False)

        # to allow execution of all metrics we remove models from memory after each metric
        unload_nli()


if __name__ == "__main__":
    main(sys.argv[1:])

"eval.py --experiment rag_gtr_k_10_llama_8b --device cuda:1 --metrics correctness_bert --overwrite true" "eval.py --experiment rag_gtr_k_10_llama_8b --device cuda:1 --metrics correctness_bert --overwrite true" "eval.py --experiment rag_gtr_k_10_saul_7b --device cuda:1 --metrics correctness_bert --overwrite true" "eval.py --experiment rag_gtr_k_10_llama_70b --device cuda:1 --metrics correctness_bert --overwrite true"
