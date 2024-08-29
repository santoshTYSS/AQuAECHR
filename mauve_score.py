import string
import sys
import mauve

from src.experiments.load_experiment import load_experiment_df


def calc_mauve(generated: list[str], target: list[str]) -> float:
    """
    This method calculates the fluency of the generated answer
    """

    target = [" ".join(t.split()[0:100]).rstrip(string.punctuation) for t in target]
    gen = [" ".join(g.split()[0:100]).rstrip(string.punctuation) for g in generated]

    out = mauve.compute_mauve(
        p_text=target,
        q_text=gen,
        max_text_length=512,
        verbose=True,
        featurize_model_name="gpt2-large",
    )

    return out.mauve


def main(args):
    if "--in" not in args:
        print("Please provide the input file using the flag --in")
        return
    e = args[args.index("--experiment") + 1]
    df, _ = load_experiment_df(e)
    answers = df["answer_no_citations"]
    generated_answers = df["generated_answer"]

    generated_answers = generated_answers.fillna("no answer")
    for i in range(len(generated_answers)):
        if len(generated_answers[i]) < 5:
            print(f"Warning: Answer {i} is too short: {generated_answers[i]}")
            generated_answers[i] = "no answer"

    mauve_score = calc_mauve(generated_answers, answers)

    print(f"MAUVE score: {(mauve_score * 100):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
