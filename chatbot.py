import os
import re
import json
import requests

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://dictionary.kubishi.com/api"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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

    "body parts": ["head", "hair", "eye", "ear", "nose", "mouth", "hand", "foot", "leg", "arm"],
    "body part": ["head", "hair", "eye", "ear", "nose", "mouth", "hand", "foot", "leg", "arm"],
    "household": ["house", "door", "bed", "blanket", "basket", "fire", "water", "food"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "verbs": ["see", "eat", "speak", "drink", "go", "come", "give", "make"],
}

STOP_WORDS = {
    "what", "is","the", "paiute", "word", "for", "how", "do", "you", "say", "in", "owens", "valley", "mean", "does", 
    "give", "me", "create", "make", "list", "related", "to", "about", "sentence", "sentences", "example", "examples", 
    "translation", "translate", "provide", "with", "show", "use", "using", "slide","slides", "lesson", "content",
    "vocabulary", "vocab", "please", "can", "a", "an", "and",
}


UNSUPPORTED_TRANSLATION_PATTERNS = [
    "how do i say",
    "how would i say",
    "translate",
    "can you translate",
]


def normalize_text(text):
    """
    Normalize common punctuation without removing Paiute characters.
    """
    return (
        text.lower()
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .strip()
    )


def extract_count(user_input, default=5, maximum=10):
    """
    Extract a requested number from the user input.
    Examples:
    - "Give me 3 example sentences" -> 3
    - "Give me one example sentence" -> 1
    - "Give me an example sentence" -> 1
    - "Give me a sentence" -> 1
    """

    text = normalize_text(user_input)

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    
    digit_match = re.search(r"\b(\d+)\b", text)

    if digit_match:
        count = int(digit_match.group(1))
        return max(1, min(count, maximum))

    for word, value in number_words.items():
        if re.search(rf"\b{word}\b", text):
            return max(1, min(value, maximum))

    single_example_patterns = [
        "an example sentence",
        "a example sentence",
        "one example sentence",
        "an example",
        "a sentence",
        "one sentence",
        "a paiute sentence",
        "one paiute sentence",
    ]

    if any(pattern in text for pattern in single_example_patterns):
        return 1

    return default

def is_unsupported_translation_request(user_input):
    """
    Detect full-sentence translation requests that the chatbot should not invent.
    """
    text = normalize_text(user_input)

    return any(pattern in text for pattern in UNSUPPORTED_TRANSLATION_PATTERNS)

def classify_request(user_input):
    """
    Decide what type of request the user is making.
    """
    text = normalize_text(user_input)

    wants_vocab = any(
        phrase in text
        for phrase in [
            "vocabulary",
            "vocab",
            "word list",
            "list of",
            "words related",
        ]
    )

    wants_slides = any(
        phrase in text
        for phrase in [
            "slide",
            "slides",
            "presentation",
            "lesson",
            "lesson plan",
        ]
    )

    wants_examples = any(
        phrase in text
        for phrase in [
            "example sentence",
            "example sentences",
            "examples",
            "sentence about",
            "sentence for",
            "sentences for",
            "use the word",
            "using the word",
        ]
    )

    wants_verified_sentence = any(
        phrase in text
        for phrase in [
            "paiute sentence",
            "verified sentence",
        ]
    )

    if wants_vocab and wants_examples:
        return "vocab_and_sentences"

    if wants_vocab and wants_slides:
        return "vocab_and_slides"

    if wants_verified_sentence or wants_examples:
        return "sentences"

    if wants_slides:
        return "slides"

    if wants_vocab:
        return "word_list"

    return "lookup"


def extract_topic(user_input):
    """
    Extract broad classroom topics like food, animals, body parts, etc.
    """
    text = normalize_text(user_input)

    topic_patterns = [
        ("body part", "body parts"),
        ("body parts", "body parts"),
        ("household", "household"),
        ("numbers", "numbers"),
        ("number", "numbers"),
        ("verbs", "verbs"),
        ("animals", "animals"),
        ("animal", "animals"),
        ("food", "food"),
        ("family", "family"),
        ("nature", "nature"),
    ]

    for phrase, topic in topic_patterns:
        if phrase in text:
            return topic
        
    remove_phrases = [
        "provide me with",
        "give me",
        "can you",
        "show me",
        "make",
        "create",
        "a list of",
        "list of",
        "word list",
        "words related to",
        "related to",
        "vocabulary",
        "vocab",
        "lesson plan",
        "lesson",
        "slides",
        "slide",
        "content",
        "about",
        "for",
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, " ")

    text = re.sub(r"\s+", " ", text).strip(" ?.!")

    return text or "general"

def extract_terms_from_multi_request(user_input):
    """
    Extract multiple search terms from a request like:
    "Give me verbs related to seeing, eating, and speaking."
    """
    text = normalize_text(user_input)

    replacements = {
        "seeing": "see",
        "eating": "eat",
        "speaking": "speak",
        "drinking": "drink",
        "going": "go",
        "coming": "come",
        "giving": "give",
        "making": "make",
    }

    terms = []

    for source, target in replacements.items():
        if source in text:
            terms.append(target)

    if terms:
        return terms

    return []

def generate_topic_words(topic, count=8):
    """
    Generate simple English dictionary search terms for broad topics.
    Uses safe fallback lists first.
    """
    topic = topic.lower().strip()

    multi_terms = extract_terms_from_multi_request(topic)
    if multi_terms:
        return multi_terms[:count]

    if topic in TOPIC_FALLBACKS:
        return TOPIC_FALLBACKS[topic][:count]

    prompt = f"""
Generate {count} simple English dictionary search terms related to this topic:

Topic: {topic}

Rules:
- Return only JSON.
- Return a JSON list of strings.
- Use common beginner vocabulary.
- Use single words when possible.
- Do not include explanations.

Example:
["water", "fish", "eat"]
"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
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
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        words = json.loads(content)

        if isinstance(words, list):
            clean_words = [
                str(word).strip()
                for word in words
                if isinstance(word, str) and word.strip()
            ]
            return clean_words[:count]

    except Exception as error:
        print(f"Topic word generation error: {error}")

    return [topic]

def unicode_words(text):
    """
    Extract words while preserving Paiute characters like ü and apostrophes.
    """
    return re.findall(r"[\wüÜ'’-]+", text, flags=re.UNICODE)

def rewrite_query(user_input):
    """
    Extract the main search term from a user's request.
    """
    text = normalize_text(user_input)

    if "food" in text:
        return "food"

    if "animal" in text or "animals" in text:
        return "animals"

    if "family" in text:
        return "family"

    if "body part" in text or "body parts" in text:
        return "body parts"

    if "household" in text:
        return "household"

    if "numbers" in text or "number" in text:
        return "numbers"

    for word in unicode_words(text):
        if any(char in word for char in ["ü", "Ü", "'", "’", "-"]):
            if word not in STOP_WORDS:
                return word.strip(" ?.!\"'")

    if "word for" in text:
        return text.split("word for")[-1].strip(" ?.!\"'")

    if "what does" in text and "mean" in text:
        return (
            text.replace("what does", "")
            .replace("mean", "")
            .strip(" ?.!\"'")
        )

    if "examples that use the word" in text:
        return text.split("examples that use the word")[-1].strip(" ?.!\"'")

    if "examples using the word" in text:
        return text.split("examples using the word")[-1].strip(" ?.!\"'")

    if "use the word" in text:
        return text.split("use the word")[-1].strip(" ?.!\"'")

    if "using the word" in text:
        return text.split("using the word")[-1].strip(" ?.!\"'")

    if "sentences for" in text:
        return text.split("sentences for")[-1].strip(" ?.!\"'")

    if "sentence for" in text:
        return text.split("sentence for")[-1].strip(" ?.!\"'")

    if "examples for" in text:
        return text.split("examples for")[-1].strip(" ?.!\"'")

    if "example sentences for" in text:
        return text.split("example sentences for")[-1].strip(" ?.!\"'")

    words = unicode_words(text)
    useful_words = [word for word in words if word not in STOP_WORDS]

    if useful_words:
        return useful_words[-1].strip(" ?.!\"'")

    return text.strip(" ?.!\"'")


def search_dictionary(query):
    """
    Search the dictionary API.
    """
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

def search_example_sentences(query, limit=20):
    """
    Search verified example sentences from the dictionary API.
    """

    try:
        response = requests.get(
            f"{BASE_URL}/search-sentences",
            params={
                "q": query,
                "limit": limit,
            },
            timeout=10,
        )

        response.raise_for_status()
        return response.json()

    except Exception as error:
        print(f"Example sentence search error: {error}")
        return None
    
def search_dictionary(query):
    """
    Search the dictionary API.
    """
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


def search_example_sentences(query, limit=20):
    """
    Search verified example sentences from the dictionary API.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/search-sentences",
            params={
                "q": query,
                "limit": limit,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except Exception as error:
        print(f"Example sentence search error: {error}")
        return None

def extract_best_entry(api_response):
    """
    Select the best dictionary entry from the API response.
    """
    if not api_response:
        return None

    results = api_response.get("results", [])

    if not results:
        return None

    entry = results[0]
    senses = entry.get("senses", [])
    first_sense = senses[0] if senses else {}

    glossary = first_sense.get("glossary")
    definition = first_sense.get("definition")

    return {
        "word": entry.get("word", "Unknown"),
        "glossary": glossary or "No glossary available",
        "definition": definition or "No definition available",
        "raw": entry,
    }

def is_reasonable_match(search_term, entry):
    """
    Protect against false positives.

    Example failure this prevents:
    - blorpa -> pishapi
    - computer -> io
    """
    if not entry:
        return False

    term = search_term.lower().strip(" ?.!\"'")
    word = entry.get("word", "").lower()
    glossary = entry.get("glossary", "").lower()
    definition = entry.get("definition", "").lower()

    if term in TOPIC_FALLBACKS:
        return True

    if term == word:
        return True

    if term in glossary:
        return True

    if term in definition:
        return True

    return False

def lookup_word(user_input):
    """
    Look up one dictionary word with weak-match protection.
    """
    search_term = rewrite_query(user_input)
    api_response = search_dictionary(search_term)
    entry = extract_best_entry(api_response)

    if not entry:
        return None

    entry["search_term"] = search_term

    if not is_reasonable_match(search_term, entry):
        return None

    return entry

def format_entry(entry):
    """
    Convert a dictionary entry into a nice Markdown response.
    """
    meaning = entry.get("glossary")

    if not meaning or meaning == "No glossary available":
        meaning = entry.get("definition", "No meaning available")

    return f"""
## Dictionary Result

**Search term:** `{entry.get("search_term", "Unknown")}`

**Paiute word:** {entry.get("word", "Unknown")}

**Meaning:** {meaning}

**Definition:** {entry.get("definition", "No definition available")}
"""

def format_api_example_sentences(example_response, search_term, entry, limit=5):
    """
    Format example sentences returned from the API.
    """
    if not example_response:
        return None

    results = example_response.get("results", [])

    if not results:
        return None

    response = f"""
## Example Sentences for `{search_term}`

**Dictionary word:** {entry.get("word", "Unknown")}

**Definition:** {entry.get("definition", "No definition available")}

These examples were retrieved from the verified dictionary example sentence database.
"""

    for index, example in enumerate(results[:limit], start=1):
        paiute = example.get("text", "No Paiute sentence provided.")
        english = example.get("translation", "No English translation provided.")

        response += f"""

### Example {index}

**Paiute:** {paiute}

**English:** {english}
"""

    return response

def build_sentences(user_input):
    """
    Retrieve verified example sentences from the dictionary sentence API.
    """
    requested_count = extract_count(user_input, default=5, maximum=10)
    search_term = rewrite_query(user_input)

    api_response = search_dictionary(search_term)
    entry = extract_best_entry(api_response)

    if not entry:
        return f"""
## No Example Sentence Found

I could not find a dictionary entry for `{search_term}`.
"""

    entry["search_term"] = search_term

    if not is_reasonable_match(search_term, entry):
        return f"""
## No Reliable Example Sentence Found

I searched for `{search_term}`, but the dictionary result did not appear to match the request closely enough.

Try asking for a specific dictionary word or a known Paiute word.
"""

    dictionary_word = entry.get("word", "")

    # First search examples using the user's query.
    example_response = search_example_sentences(search_term, limit=requested_count)
    formatted_examples = format_api_example_sentences(
        example_response=example_response,
        search_term=search_term,
        entry=entry,
        limit=requested_count,
    )

    if formatted_examples:
        return formatted_examples

    # If no examples are found, try the Paiute dictionary word.
    if dictionary_word:
        example_response = search_example_sentences(dictionary_word, limit=requested_count)
        formatted_examples = format_api_example_sentences(
            example_response=example_response,
            search_term=search_term,
            entry=entry,
            limit=requested_count,
        )

        if formatted_examples:
            return formatted_examples

    return f"""
## No Verified Example Sentence Found

I found the dictionary entry, but I could not find verified example sentences from the sentence API.

**Search term:** `{search_term}`

**Dictionary word:** {dictionary_word}

**Definition:** {entry.get("definition", "No definition available")}
"""

def build_word_list(topic):
    """
    Build a vocabulary list using dictionary API lookups.
    """
    search_terms = generate_topic_words(topic)
    entries = []

    for term in search_terms:
        api_response = search_dictionary(term)
        entry = extract_best_entry(api_response)

        if not entry:
            continue

        entry["search_term"] = term

        if not is_reasonable_match(term, entry):
            continue

        entries.append(entry)

    if not entries:
        return f"""
## No Vocabulary Found

I could not find reliable vocabulary for `{topic}`.
"""

    response = f"## Vocabulary List: {topic.title()}\n\n"
    response += (
        "These entries are retrieved dictionary matches. "
        "Some may be partial or culturally specific matches rather than exact beginner vocabulary.\n\n"
    )
    response += f"Generated search terms: `{', '.join(search_terms)}`\n\n"

    for entry in entries:
        meaning = entry.get("glossary")

        if not meaning or meaning == "No glossary available":
            meaning = entry.get("definition", "No meaning available")

        response += f"- **{entry['word']}** — {meaning}\n"
        response += f"  - Search term: `{entry['search_term']}`\n"
        response += f"  - Definition: {entry['definition']}\n\n"

    return response

def build_slides(user_input):
    """
    Create a simple lesson outline from retrieved vocabulary only.
    """
    topic = extract_topic(user_input)
    slide_count = extract_count(user_input, default=4, maximum=8)

    vocab = build_word_list(topic)

    if "## No Vocabulary Found" in vocab:
        return f"""
## No Slides Created

I could not find reliable vocabulary for `{topic}`.
"""

    prompt = f"""
Create a simple {slide_count}-slide lesson outline using ONLY this retrieved vocabulary content:

{vocab}

Rules:
- Do not invent new Paiute words.
- Use only the Paiute vocabulary shown in the retrieved content.
- Keep it classroom-friendly.
- If a vocabulary match seems awkward or culturally specific, mention that it may be a partial dictionary match.
- Include a title, learning goal, vocabulary slide, practice activity, and review question.
"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create simple educational lesson outlines from retrieved "
                    "dictionary vocabulary only. You do not invent Paiute words."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()

def build_vocab_and_sentences(user_input):
    """
    Handle combined requests like:
    'Give me food vocabulary and example sentences.'
    """
    topic = extract_topic(user_input)

    vocab_content = build_word_list(topic)
    sentence_content = build_sentences(topic)

    return f"""
{vocab_content}

---

{sentence_content}
"""


def build_vocab_and_slides(user_input):
    """
    Handle combined requests like:
    'Give me animal vocabulary and slide content.'
    """
    topic = extract_topic(user_input)

    vocab_content = build_word_list(topic)
    slide_content = build_slides(user_input)

    return f"""
{vocab_content}

---

{slide_content}
"""


def safe_translation_response():
    """
    Response for unsupported full-sentence translation requests.
    """
    return """
## Translation Not Available

I cannot safely create a new Paiute sentence from English unless that sentence appears in the verified source data.

I can still help by:
- looking up individual dictionary words,
- retrieving verified example sentences,
- creating vocabulary lists from dictionary entries,
- or creating slide content using retrieved vocabulary.
"""


def handle_ambiguous_sentence_check(user_input):
    """
    Handle questions like:
    'Is this a real Paiute sentence?'
    where no sentence is actually provided.
    """
    text = normalize_text(user_input)

    if "is this a real paiute sentence" in text and '"' not in text:
        return """
## Sentence Needed

Please provide the Paiute sentence you want checked.

I can help compare it against verified example sentences or dictionary entries, but I need the actual sentence first.
"""

    return None

def process_input(user_input):
    """
    Main function called by app.py or main.py.
    """
    ambiguous_response = handle_ambiguous_sentence_check(user_input)

    if ambiguous_response:
        return ambiguous_response

    if is_unsupported_translation_request(user_input):
        return safe_translation_response()

    intent = classify_request(user_input)

    if intent == "vocab_and_sentences":
        return build_vocab_and_sentences(user_input)

    if intent == "vocab_and_slides":
        return build_vocab_and_slides(user_input)

    if intent == "word_list":
        topic = extract_topic(user_input)
        return build_word_list(topic)

    if intent == "sentences":
        return build_sentences(user_input)

    if intent == "slides":
        return build_slides(user_input)

    entry = lookup_word(user_input)

    if entry:
        return format_entry(entry)

    search_term = rewrite_query(user_input)

    return f"""
## No Reliable Dictionary Result Found

I searched for `{search_term}`, but I could not find a reliable matching dictionary entry.

Try asking for:
- a specific English word,
- a known Paiute word,
- a vocabulary list by topic,
- or verified example sentences.
"""


def evaluate_query(user_input):
    """
    Helper function for Week 5 evaluation.
    This lets you inspect routing and query extraction.
    """
    intent = classify_request(user_input)

    if intent in ["word_list", "slides", "vocab_and_sentences", "vocab_and_slides"]:
        topic_or_term = extract_topic(user_input)
    else:
        topic_or_term = rewrite_query(user_input)

    response = process_input(user_input)

    return {
        "query": user_input,
        "intent": intent,
        "topic_or_term": topic_or_term,
        "response": response,
    }


if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question, or type 'quit': ")

        if user_input.lower().strip() in ["quit", "exit"]:
            break

        print(process_input(user_input))