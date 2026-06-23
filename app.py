import streamlit as st

from chatbot import process_input

st.set_page_config(
    page_title="Owens Valley Paiute Chatbot",
    page_icon="💬",
)

st.title("Owens Valley Paiute Chatbot")

st.write(
    "Ask a question about Owens Valley Paiute vocabulary."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "Example: What is the Paiute word for water?"
)


if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching the dictionary..."):
        result = process_input(prompt)

    if result:
        response = f"""
Searched for: `{result["search_term"]}`

### {result["word"]}

**Gloss:** {result["gloss"]}

**Definition:** {result["definition"]}
"""
    else:
        response = (
            "I could not find a matching dictionary entry. "
            "Try asking with a simpler word, like `water`, `dog`, or `food`."
        )

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
    