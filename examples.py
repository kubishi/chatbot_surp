import json
import re
from pathlib import Path


EXAMPLES_PATH = Path("examples.json")


def load_examples():
    """
    Load verified example sentences from examples.json.
    """

    if not EXAMPLES_PATH.exists():
        return []

    with open(EXAMPLES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    """
    Lowercase and remove punctuation for easier matching.
    Keeps basic Paiute characters and apostrophes.
    """

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Zāēīōūü' -]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_example_sentences(search_terms, max_examples=5):
    """
    Search verified example sentences using English and Paiute search terms.

    Example:
    search_terms = ["water", "paya"]
    """

    examples = load_examples()

    if isinstance(search_terms, str):
        search_terms = [search_terms]

    normalized_terms = [
        normalize_text(term)
        for term in search_terms
        if term
    ]

    matches = []

    for example in examples:
        paiute = normalize_text(example.get("paiute", ""))
        english = normalize_text(example.get("english", ""))

        keywords = example.get("keywords", [])
        keywords_text = normalize_text(" ".join(keywords))

        searchable_text = f"{paiute} {english} {keywords_text}"

        if any(term in searchable_text for term in normalized_terms):
            matches.append(example)

    return matches[:max_examples]