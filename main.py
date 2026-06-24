import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BASE_URL = "https://dictionary.kubishi.com/api"

DEBUG = False


# =========================
# LOGGING (3-stage trace)
# =========================
def log(step, data):
    if DEBUG:
        print(f"\n[{step}]")
        print(data)


# =========================
# 1. QUERY EXTRACTION
# =========================
def rewrite_query(user_input):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the main English noun or concept. "
                    "Return ONLY ONE word. No punctuation. No explanation."
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0
    )

    keyword = response.choices[0].message.content.strip().lower()

    # safety cleanup
    if len(keyword.split()) > 1:
        keyword = keyword.split()[0]

    log("QUERY", keyword)

    return keyword


# =========================
# 2. DICTIONARY SEARCH
# =========================
def search_dictionary(query):
    response = requests.get(
        f"{BASE_URL}/search",
        params={"q": query}
    )
    return response.json()


# =========================
# 3. ENTRY SELECTION (NO HARD FAILS)
# =========================
def extract_best_entry(api_response, keyword):

    results = api_response.get("results", [])

    if not results:
        return None

    keyword = keyword.lower()

    # try best match first
    for entry in results:
        word = entry.get("word", "").lower()

        if keyword in word:
            senses = entry.get("senses", [])
            sense = senses[0] if senses else {}

            return {
                "word": entry.get("word", ""),
                "glossary": sense.get("glossary", "Not available"),
                "definition": sense.get("definition", "Not available")
            }

    # fallback: still return something valid
    entry = results[0]
    senses = entry.get("senses", [])
    sense = senses[0] if senses else {}

    return {
        "word": entry.get("word", ""),
        "glossary": sense.get("glossary", "Not available"),
        "definition": sense.get("definition", "Not available")
    }


# =========================
# 4. GENERATION (FULL CONTEXT GROUNDED)
# =========================
def generate_response(user_input, entry, raw_results):

    prompt = f"""
You are a Paiute language assistant.

USER QUESTION:
{user_input}

SELECTED ENTRY:
{entry}

FULL DICTIONARY RESULTS:
{raw_results}

RULES:
- Use ONLY the dictionary data provided
- NEVER invent Paiute words
- If multiple entries exist, choose the most relevant one
- If none are perfect, explain closest meaning
- NEVER say "no match found"
- Always stay grounded in the provided data
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict, dictionary-grounded language assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# =========================
# 5. PIPELINE CONTROLLER
# =========================
def process_input(user_input):

    # Step 1: keyword
    query = rewrite_query(user_input)
    log("DICTIONARY QUERY", query)

    # Step 2: search
    data = search_dictionary(query)

    raw_results = data.get("results", [])
    log("RAW RESULTS", raw_results)

    # Step 3: select entry (always returns something if possible)
    entry = extract_best_entry(data, query)
    log("SELECTED ENTRY", entry)

    # Step 4: generate using FULL context
    if not raw_results:
        return "No dictionary results found."

    return generate_response(user_input, entry, raw_results)


# =========================
# 6. OUTPUT FORMAT
# =========================
def format_output(text):
    return f"""
━━━━━━━━━━━━━━━━━━
📘 Paiute Assistant
━━━━━━━━━━━━━━━━━━
{text}
━━━━━━━━━━━━━━━━━━
""".strip()


# =========================
# 7. MAIN LOOP
# =========================
def main():
    print("Owens Valley Paiute Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = process_input(user_input)

        print("\n" + format_output(response) + "\n")


if __name__ == "__main__":
    main()