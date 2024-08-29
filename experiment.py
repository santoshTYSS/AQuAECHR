import argparse
from src.experiments.base import base_completion_loop
from src.experiments.llatrieval import llatrieval_loop
from src.experiments.load_experiment import Experiment
from src.experiments.post_hoc import post_hoc_loop
from src.experiments.rag import rag_loop
from src.experiments.rarr import rarr_loop
from src.llms.groq import Llama70B, Llama8B
from src.llms.mistral import Mistral, MistralDeepInfra
from src.llms.saul import Saul
from src.retrievers.bm25 import BM25
from src.retrievers.gtr_t5 import GTR


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment to run",
        choices=[e.value for e in Experiment],
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the experiment on",
        default="cpu",
    )

    args = parser.parse_args()

    print(f"Running experiment {args.experiment}")
    match args.experiment:
        # Base Experiments
        case Experiment.BASE_LLAMA_8B:
            base_completion_loop(
                llm=Llama8B(),
                experiment=Experiment.BASE_LLAMA_8B,
            )
        case Experiment.BASE_MISTRAL_7B:
            base_completion_loop(
                llm=Mistral(device=args.device),
                experiment=Experiment.BASE_MISTRAL_7B,
            )
        case Experiment.BASE_SAUL_7B:
            base_completion_loop(
                llm=Saul(device=args.device),
                experiment=Experiment.BASE_SAUL_7B,
            )
        case Experiment.BASE_LLAMA_70B:
            base_completion_loop(
                llm=Llama70B(),
                experiment=Experiment.BASE_LLAMA_70B,
            )

        # RAG Experiments
        case Experiment.RAG_GTR_k_10_LLAMA_8B:
            rag_loop(
                llm=Llama8B(),
                retriever=GTR(),
                experiment=Experiment.RAG_GTR_k_10_LLAMA_8B,
                k=10,
            )
        case Experiment.RAG_GTR_k_10_MISTRAL_7B:
            rag_loop(
                llm=Mistral(device=args.device),
                retriever=GTR(),
                experiment=Experiment.RAG_GTR_k_10_MISTRAL_7B,
                k=10,
            )
        case Experiment.RAG_GTR_k_10_SAUL_7B:
            rag_loop(
                llm=Saul(device=args.device),
                retriever=GTR(),
                experiment=Experiment.RAG_GTR_k_10_SAUL_7B,
                k=10,
            )
        case Experiment.RAG_GTR_k_10_LLAMA_70B:
            rag_loop(
                llm=Llama70B(),
                retriever=GTR(),
                experiment=Experiment.RAG_GTR_k_10_LLAMA_70B,
                k=10,
            )

        # LLATRIEVAL Experiments
        case Experiment.LLATRIEVAL_GTR_k_10_LLAMA_8B:
            llatrieval_loop(
                llm=Llama8B(),
                retriever=GTR(),
                experiment=Experiment.LLATRIEVAL_GTR_k_10_LLAMA_8B,
                k=10,
            )
        case Experiment.LLATRIEVAL_GTR_k_10_MISTRAL_7B:
            llatrieval_loop(
                llm=MistralDeepInfra(),
                retriever=GTR(),
                experiment=Experiment.LLATRIEVAL_GTR_k_10_MISTRAL_7B,
                k=10,
            )
        case Experiment.LLATRIEVAL_GTR_k_10_SAUL_7B:
            llatrieval_loop(
                llm=Saul(device=args.device),
                retriever=GTR(),
                experiment=Experiment.LLATRIEVAL_GTR_k_10_SAUL_7B,
                k=10,
            )
        case Experiment.LLATRIEVAL_GTR_k_10_LLAMA_70B:
            llatrieval_loop(
                llm=Llama70B(),
                retriever=GTR(),
                experiment=Experiment.LLATRIEVAL_GTR_k_10_LLAMA_70B,
                k=10,
            )

        # Post Hoc Experiments
        case Experiment.POST_HOC_LLAMA_8B:
            post_hoc_loop(
                retriever=GTR(),
                experiment=Experiment.POST_HOC_LLAMA_8B,
                threshold=0.5,
            )
        case Experiment.POST_HOC_MISTRAL_7B:
            post_hoc_loop(
                retriever=GTR(),
                experiment=Experiment.POST_HOC_MISTRAL_7B,
                threshold=0.5,
            )
        case Experiment.POST_HOC_SAUL_7B:
            post_hoc_loop(
                retriever=GTR(),
                experiment=Experiment.POST_HOC_SAUL_7B,
                threshold=0.5,
            )
        case Experiment.POST_HOC_LLAMA_70B:
            post_hoc_loop(
                retriever=GTR(),
                experiment=Experiment.POST_HOC_LLAMA_70B,
                threshold=0.5,
            )

        # RARR Experiments
        case Experiment.RARR_LLAMA_8B:
            rarr_loop(
                llm=Llama8B(),
                retriever=GTR(),
                experiment=Experiment.RARR_LLAMA_8B,
            )
        case Experiment.RARR_MISTRAL_7B:
            rarr_loop(
                llm=MistralDeepInfra(),
                retriever=GTR(),
                experiment=Experiment.RARR_MISTRAL_7B,
            )
        case Experiment.RARR_SAUL_7B:
            rarr_loop(
                llm=Saul(device=args.device),
                retriever=GTR(),
                experiment=Experiment.RARR_SAUL_7B,
            )
        case Experiment.RARR_LLAMA_70B:
            rarr_loop(
                llm=Llama70B(),
                retriever=GTR(),
                experiment=Experiment.RARR_LLAMA_70B,
            )
        case Experiment.RAG_BM25_k_10_LLAMA_70B:
            rag_loop(
                llm=Llama70B(),
                retriever=BM25(),
                experiment=Experiment.RAG_BM25_k_10_LLAMA_70B,
                k=10,
            )
        case _:
            raise ValueError(f"Experiment {args.experiment} not implemented")


if __name__ == "__main__":
    main()
