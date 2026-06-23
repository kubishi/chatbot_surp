import os
import re
import json
import requests
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

BASE_URL = "https://dictionary.kubishi.com/api"


def classify_request(user_input):
    text = user_input.lower()

    if any(word in text for word in ["list", "words related", "vocabulary", "vocab"]):
        return "word_list"

    if any(word in text for word in ["sentence", "sentences", "use in a sentence"]):
        return "sentences"

    if any(word in text for word in ["slide", "slides", "presentation", "lesson"]):
        return "slides"

    return "lookup"


def extract_topic(user_input):
    text = user_input.lower().strip()

    for phrase in ["related to", "about", "for"]:
        if phrase in text:
            return text.split(phrase)[-1].strip(" ?.!")

    return text.strip(" ?.!")



def generate_topic_words(topic, count=8):
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
        "gloss": first_sense.get("gloss", "No gloss available"),
        "definition": first_sense.get("definition", "No definition available"),
    }


def lookup_word(user_input):
    search_term = rewrite_query(user_input)
    api_response = search_dictionary(search_term)
    entry = extract_best_entry(api_response)

    if entry:
        entry["search_term"] = search_term
        return entry

    return None


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
        response += f"- **{entry['word']}** — {entry['gloss']}\n"
        response += f"  - Search term: `{entry['search_term']}`\n"
        response += f"  - Definition: {entry['definition']}\n\n"

    return response


def build_sentences(user_input):
    entry = lookup_word(user_input)

    if not entry:
        return None

    prompt = f"""
Create 3 beginner-friendly English practice sentences for this dictionary entry.

Word: {entry["word"]}
Gloss: {entry["gloss"]}
Definition: {entry["definition"]}

Rules:
- Do not invent Paiute grammar.
- Use English sentences.
- Keep the sentences simple.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You create careful beginner language-learning materials.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return f"""
## Example Sentences

**Word:** {entry["word"]}

**Meaning:** {entry["gloss"]}

{response.choices[0].message.content}
"""


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

Keep it classroom-friendly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You create simple educational slide outlines.",
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

    if intent == "word_list":
        topic = extract_topic(user_input)
        content = build_word_list(topic)

    elif intent == "sentences":
        content = build_sentences(user_input)

    elif intent == "slides":
        content = build_slides(user_input)

    else:
        entry = lookup_word(user_input)

        if entry:
            content = f"""
Searched for: `{entry["search_term"]}`

### {entry["word"]}

**Gloss:** {entry["gloss"]}

**Definition:** {entry["definition"]}
"""
        else:
            content = None

    if not content:
        content = (
            "I could not find enough dictionary information for that request. "
            "Try asking with a simpler word or topic, like `water`, `animals`, `food`, or `family`."
        )

    return {
        "type": "formatted_text",
        "content": content,
    }