import pandas as pd

from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.llms.llm import LLM


def base_completion_loop(llm: LLM, experiment: Experiment):
    df, path = load_experiment_df(experiment)
    for i, row in df.iterrows():
        if Column.GENERATED_ANSWER in row and pd.notnull(row[Column.GENERATED_ANSWER]):
            print("Skipping row", i, "as it already has a response")
            continue
        print("Processing row", i)
        question = row["question"]
        prompt = f"You are an ECHR legal expert tasked to answer the following question:\nQuestion: {question}\nAnswer:"
        response = llm.infer_completion(prompt)
        print(f"Question: {question}")
        print(f"Response: {response}")
        print()
        df.at[i, Column.GENERATED_ANSWER] = response
        df.to_csv(path, index=False)
