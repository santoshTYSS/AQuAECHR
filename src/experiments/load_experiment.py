from enum import Enum
import os
import csv

import pandas as pd

from src.constants import ECHR_QA_CSV_PATH


"eval.py --experiment llatrieval_gtr_k_10_llama_70b --device cuda:0" "eval.py --experiment rarr_llama_8b --device cuda:0" "eval.py --experiment rarr_mistral_7b --device cuda:0"

"python experiment.py --experiment rag_gtr_k_10_llama_70b --device cuda:0"


# python experiment.py --experiment rag_gtr_k_10_llama_70b --device cuda:0
# python eval.py --experiment rag_gtr_k_10_llama_70b --device cuda:0
# python eval.py --experiment base_llama_8b --device cuda:0 --metric "correctness_bert, correctness_rouge, claim_recall, citation_faithfulness, citation_similarity_em, citation_similarity_nli"
class Experiment(str, Enum):
    # Base Experiments
    BASE_LLAMA_8B = "base_llama_8b"
    BASE_MISTRAL_7B = "base_mistral_7b"
    BASE_SAUL_7B = "base_saul_7b"
    BASE_LLAMA_70B = "base_llama_70b"
    # RAG Experiments
    RAG_GTR_k_10_LLAMA_8B = "rag_gtr_k_10_llama_8b"
    RAG_GTR_k_10_MISTRAL_7B = "rag_gtr_k_10_mistral_7b"
    RAG_GTR_k_10_SAUL_7B = "rag_gtr_k_10_saul_7b"
    RAG_GTR_k_10_LLAMA_70B = "rag_gtr_k_10_llama_70b"
    # LLATRIEVAL Experiments
    LLATRIEVAL_GTR_k_10_LLAMA_8B = "llatrieval_gtr_k_10_llama_8b"
    LLATRIEVAL_GTR_k_10_MISTRAL_7B = "llatrieval_gtr_k_10_mistral_7b"
    LLATRIEVAL_GTR_k_10_SAUL_7B = "llatrieval_gtr_k_10_saul_7b"
    LLATRIEVAL_GTR_k_10_LLAMA_70B = "llatrieval_gtr_k_10_llama_70b"
    # Post Hoc Experiments
    POST_HOC_LLAMA_8B = "post_hoc_llama_8b"
    POST_HOC_MISTRAL_7B = "post_hoc_mistral_7b"
    POST_HOC_SAUL_7B = "post_hoc_saul_7b"
    POST_HOC_LLAMA_70B = "post_hoc_llama_70b"
    # RARR Experiments
    RARR_LLAMA_8B = "rarr_llama_8b"
    RARR_MISTRAL_7B = "rarr_mistral_7b"
    RARR_SAUL_7B = "rarr_saul_7b"
    RARR_LLAMA_70B = "rarr_llama_70b"
    # RAG with BM25 Experiment
    RAG_BM25_k_10_LLAMA_70B = "rag_bm25_k_10_llama_70b"


def load_experiment_df(e: Experiment):
    path = "data/e_" + e + ".csv"
    return (
        pd.read_csv(path) if os.path.exists(path) else pd.read_csv(ECHR_QA_CSV_PATH)
    ), path
