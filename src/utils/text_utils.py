import re
import spacy


nlp = spacy.load("en_core_web_trf")


def remove_unwanted_citations_at_the_end(sentences: list[str]):
    """
    LLMs often add a list of the used citations at the end of the response. We try to remove these citations.
    We remove all sentences at the end that do not end in a "." or start with a citation.
    """

    while sentences:
        if (
            not sentences[-1].strip().endswith(".")
            or sentences[-1].strip().startswith("[")
            or "citations:" in sentences[-1].lower()
        ):
            sentences.pop()
        else:
            break

    return sentences


def get_sentences(response: str):
    """
    We split the response into sentences using spaCy.
    """
    initial_sentences = response.split("\n")  # an llm does not randomly add newlines
    sentences = []
    for sentence in initial_sentences:
        if sentence.strip():
            doc = nlp(sentence)
            sentences.extend([s.text for s in doc.sents if s.text.strip()])

    sentences = remove_unwanted_citations_at_the_end(sentences)
    return sentences


def remove_citations(sentence: str):
    """
    Utility functions for extracting / removing citations from the response
    """
    pattern = r" \[.*?\]"
    return re.sub(pattern, "", sentence)


def get_citations(sentence: str) -> list[int]:
    """
    Extract citations from a sentence in the following forms:
    - Doc 1
    - Doc 1, 2
    - Doc 2-5
    - Doc 2, 3, and 4
    - Doc 1 and 2
    - [1]
    - [2, 3]
    - [2 and 3]
    - [3-5]

    Args:
    sentence (str): The input sentence containing citations.

    Returns:
    list of int: The list of extracted document numbers.
    """
    # Normalize the sentence by replacing variations of "and" with commas
    sentence = sentence.replace(", and", ",")
    sentence = sentence.replace(" and", ",")

    # Patterns to match the different citation styles
    patterns = [
        r"Doc (\d+(?:-\d+)?(?:, \d+)*)",  # Doc 1, Doc 1, 2, Doc 2-5, Doc 2, 3, and 4, Doc 1 and 2
        r"\[(\d+(?:-\d+)?(?:, \d+)*)\]",  # [1], [2, 3], [2 and 3], [3-5]
    ]

    citations = set()

    for pattern in patterns:
        matches = re.findall(pattern, sentence)
        for match in matches:
            parts = match.split(", ")
            for part in parts:
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    citations.update(range(start, end + 1))
                else:
                    citations.add(int(part))

    return sorted(citations)
