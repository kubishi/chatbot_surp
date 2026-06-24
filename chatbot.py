import os
import re
import json
import requests

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://dictionary.kubishi.com/api"

def get_openai_api_key():

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


api_key = get_openai_api_key()

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Add it to a .env file or to .streamlit/secrets.toml."
    )


client = OpenAI(api_key=api_key)

TOPIC_FALLBACKS = {
    "animals": ["dog", "horse", "rabbit", "fish", "bird", "deer"],
    "animal": ["dog", "horse", "rabbit", "fish", "bird", "deer"],
    "food": ["food", "water", "meat", "fish", "eat", "drink"],
    "family": ["mother", "father", "daughter", "son", "child", "woman", "man"],
    "nature": ["water", "mountain", "tree", "sun", "moon", "fire", "wind"],
}

def classify_request(user_input):
    text = user_input.lower()

    if any(phrase in text for phrase in ["paiute sentence", "full sentence", "translated sentence"]
    ):
        return "verified_sentence"
    
    if any(word in text for word in ["list", "words related", "vocabulary", "vocab"]):
        return "word_list"

    if any(word in text for word in ["sentence", "sentences", "use in a sentence"]):
        return "sentences"

    if any(word in text for word in ["slide", "slides", "presentation", "lesson"]):
        return "slides"

    return "lookup"


def extract_topic(user_input):
    text = user_input.lower().strip()

    remove_phrases = [
        "provide me with",
        "give me",
        "make",
        "create",
        "list",
        "a list of",
        "vocabulary",
        "words",
        "word list",
        "related to",
        "about",
        "lesson",
        "slides",
        "slide",
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, "")

    text = text.strip(" ?.!")
    
    if "animal" in text:
        return "animals"
    if "food" in text:
        return "food"
    if "family" in text:
        return "family"
    if "nature" in text:
        return "nature"

    return text

def generate_topic_words(topic, count=8):

    topic = topic.lower().strip()

    if topic in TOPIC_FALLBACKS:
        return TOPIC_FALLBACKS[topic][:count]
    
    prompt = f"""
Generate {count} simple English dictionary search terms related to the topic "{topic}".

Rules:
- Return only JSON.
- Use common beginner vocabulary.
- Use single words when possible.
- Do not include explanations.

Example:
["water", "fish", "eat"]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You generate simple English vocabulary search terms.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content.strip()
        words = json.loads(content)

        if isinstance(words, list):
            return words[:count]

    except Exception as error:
        print(f"Topic word generation error: {error}")

    return [topic]


def rewrite_query(user_input):
    text = user_input.lower().strip()

    if "word for" in text:
        return text.split("word for")[-1].strip(" ?.!")

    if "say" in text:
        return text.split("say")[-1].strip(" ?.!")

    if "what does" in text and "mean" in text:
        return (
            text.replace("what does", "")
            .replace("mean", "")
            .strip(" ?.!'\"")
        )

    words = re.findall(r"[a-zA-Z]+", text)

    stop_words = {
        "what", "is", "the", "paiute", "word", "for", "how", "do",
        "you", "say", "in", "owens", "valley", "mean", "does",
        "give", "me", "create", "make", "list", "related", "to",
        "about", "sentence", "sentences", "slides", "lesson"
    }

    useful_words = [word for word in words if word not in stop_words]

    if useful_words:
        return useful_words[-1]

    return text


def search_dictionary(query):
    try:
        response = requests.get(
            f"{BASE_URL}/search",
            params={"q": query},
            timeout=10,
        )

        response.raise_for_status()
        return response.json()

    except Exception as error:
        print(f"Dictionary search error: {error}")
        return None


def extract_best_entry(api_response):
    if not api_response:
        return None

    results = api_response.get("results", [])

    if not results:
        return None

    entry = results[0]
    senses = entry.get("senses", [])
    first_sense = senses[0] if senses else {}

    return {
        "word": entry.get("word", "Unknown"),
        "glossary": first_sense.get("glossary", "No glossary available"),
        "definition": first_sense.get("definition", "No definition available"),
    }

def extract_examples_from_entry(api_response):
    if not api_response:
        return []

    results = api_response.get("results", [])
    examples = []

    possible_example_fields = [
        "examples",
        "example_sentences",
        "sentences",
        "usage_examples",
    ]

    for entry in results:
        for field in possible_example_fields:
            if field in entry and isinstance(entry[field], list):
                examples.extend(entry[field])

        senses = entry.get("senses", [])

        for sense in senses:
            for field in possible_example_fields:
                if field in sense and isinstance(sense[field], list):
                    examples.extend(sense[field])

    return examples


def normalize_example(example):
    if isinstance(example, str):
        return {
            "paiute": example,
            "english": "No English translation provided.",
        }

    if isinstance(example, dict):
        paiute = (
            example.get("paiute")
            or example.get("sentence")
            or example.get("text")
            or example.get("ovp")
            or example.get("source")
            or example.get("paiute_sentence")
            or "No Paiute sentence provided."
        )

        english = (
            example.get("english")
            or example.get("translation")
            or example.get("glossary")
            or example.get("meaning")
            or example.get("english_translation")
            or "No English translation provided."
        )

        return {
            "paiute": paiute,
            "english": english,
        }

    return {
        "paiute": "No Paiute sentence provided.",
        "english": "No English translation provided.",
    }

def format_entry(entry):
    """
    Convert a dictionary entry into a nice Markdown response.
    """

    return f"""
## Dictionary Result

**Search term:** `{entry.get("search_term", "Unknown")}`

**Paiute word:** {entry.get("word", "Unknown")}

**Meaning:** {entry.get("glossary", "No glossary available")}

**Definition:** {entry.get("definition", "No definition available")}
"""

def lookup_word(user_input):
    search_term = rewrite_query(user_input)
    api_response = search_dictionary(search_term)
    entry = extract_best_entry(api_response)

    if entry:
        entry["search_term"] = search_term
        return entry

    return None

def retrieve_example_sentence(user_input):
    search_term = rewrite_query(user_input)
    api_response = search_dictionary(search_term)

    entry = extract_best_entry(api_response)
    examples = extract_examples_from_entry(api_response)

    if not entry:
        return None

    if not examples:
        return {
            "word": entry["word"],
            "glossary": entry["glossary"],
            "definition": entry["definition"],
            "search_term": search_term,
            "paiute_sentence": None,
            "english_translation": None,
        }

    normalized = [normalize_example(example) for example in examples]

    best_example = normalized[0]

    return {
        "word": entry["word"],
        "glossary": entry["glossary"],
        "definition": entry["definition"],
        "search_term": search_term,
        "paiute_sentence": best_example["paiute"],
        "english_translation": best_example["english"],
    }


def explain_verified_sentence(sentence_data):
    if not sentence_data:
        return None

    if not sentence_data["paiute_sentence"]:
        return f"""
## No Example Sentence Found

**Search term:** `{sentence_data["search_term"]}`

**Dictionary word:** {sentence_data["word"]}

**Meaning:** {sentence_data["glossary"]}

**Definition:** {sentence_data["definition"]}

I found the dictionary entry, but I did not find a verified example sentence attached to this entry.
"""

    prompt = f"""
You are helping explain a verified Owens Valley Paiute example sentence.

Use ONLY the information below.
Do not invent new Paiute words.
Do not create a new Paiute sentence.
Do not claim grammar rules unless they are directly visible from the sentence.

Dictionary entry:
Search term: {sentence_data["search_term"]}
Paiute word: {sentence_data["word"]}
Meaning: {sentence_data["glossary"]}
Definition: {sentence_data["definition"]}

Verified example sentence:
Paiute: {sentence_data["paiute_sentence"]}
English: {sentence_data["english_translation"]}
Explain the Paiute sentence in detail, including:
- Word-by-word breakdown
- Grammatical structure

"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain retrieved Owens Valley Paiute examples carefully. "
                    "You do not invent grammar or new sentences."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content

def format_verified_sentence(sentence_data):
    return f"""
## Verified Example Sentence

**Search term:** `{sentence_data["search_term"]}`

**Dictionary word:** {sentence_data["word"]}

**Meaning:** {sentence_data["glossary"]}

**Paiute sentence:** {sentence_data["paiute_sentence"]}

**English translation:** {sentence_data["english_translation"]}

This sentence was retrieved from the source data.
"""

def build_word_list(topic):
    search_terms = generate_topic_words(topic)
    entries = []

    for term in search_terms:
        api_response = search_dictionary(term)
        entry = extract_best_entry(api_response)

        if entry:
            entry["search_term"] = term
            entries.append(entry)

    if not entries:
        return None

    response = f"## Vocabulary List: {topic.title()}\n\n"
    response += f"Generated search terms: `{', '.join(search_terms)}`\n\n"

    for entry in entries:
        response += f"- **{entry['word']}** — {entry['glossary']}\n"
        response += f"  - Search term: `{entry['search_term']}`\n"
        response += f"  - Definition: {entry['definition']}\n\n"

    return response


def build_sentences(user_input):
    sentence_data = retrieve_example_sentence(user_input)

    if not sentence_data:
        return None

    if sentence_data["paiute_sentence"]:
        return format_verified_sentence(sentence_data)


def build_slides(user_input):
    topic = extract_topic(user_input)
    vocab = build_word_list(topic)

    if not vocab:
        return None

    prompt = f"""
Create a simple 4-slide lesson outline using this vocabulary content:

{vocab}

Format:
Slide 1: Title and learning goal
Slide 2: Vocabulary words
Slide 3: Practice activity
Slide 4: Review question

Rules:
- Do not invent new Paiute words.
- Use only the vocabulary shown in the retrieved content.
- Keep it classroom-friendly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You create simple educational slide outlines from retrieved vocabulary only..",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content


def process_input(user_input):
    intent = classify_request(user_input)

    if intent == "verified_sentence":
        sentence_data = retrieve_example_sentence(user_input)
        content = explain_verified_sentence(sentence_data)

    elif intent == "word_list":
        topic = extract_topic(user_input)
        content = build_word_list(topic)

    elif intent == "sentences":
        content = build_sentences(user_input)

    elif intent == "slides":
        content = build_slides(user_input)

    else:
        entry = lookup_word(user_input)

        if entry:
                content = format_entry(entry)