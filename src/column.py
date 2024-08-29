from enum import Enum


class Column(str, Enum):
    GENERATED_ANSWER = "generated_answer"
    GENERATED_CITATIONS = "generated_citations"
    TARGET_CITATIONS = "citations"
    QUESTION = "question"
    TARGET_ANSWER = "answer"
    CITATIONS = "citations"
    CORRECTNESS_BERT = "correctness_bert"
    CORRECTNESS_ROUGE = "correctness_rouge"
    CLAIM_RECALL = "claim_recall"
    CITATION_FAITHFULNESS = "citation_faithfulness"
    CITATION_SIMILARITY_EM = "citation_similarity_em"
    CITATION_SIMILARITY_NLI = "citation_similarity_nli"
