# Chatbot Project

**Project Description and Goals**

*Background*: Open with the crisis facing endangered languages: thousands at risk of disappearing, and with them irreplaceable cultural knowledge. Introduce the specific language you'll be working with (Owens Valley Paiute). Explain that language teachers are often have limited time, and that producing teaching materials is a major bottleneck. Then bridge to how LLMs and retrieval-augmented generation (RAG) offer a new approach.  Rather than relying on the LLM's own (limited or incorrect) knowledge of the language, RAG lets the system ground its outputs in verified dictionary entries, example sentences, and grammatical notes.

*Objectives*: The overall objective is to build a prototype chatbot that assists language teachers in generating pedagogical materials (word lists organized by theme, example sentences with translations, and slide content) grounded in an existing curated dictionary and linguistic resources.

Timeline:

- [ ] Week 1: Literature review on RAG and MCP; set up development environment; experiment with dictionary.kubishi.com API.
  - [ ] Follow this: https://gist.github.com/jaredraycoleman/3e7ed113bd15715831eb0e32537577d7
  - [ ] Add this to reading (after 2): https://www.anthropic.com/news/model-context-protocol
- [ ] Week 2: Implement the retrieval pipeline; run initial retrieval experiments to test that relevant entries are being surfaced for sample queries.
- [ ] Week 3: Connect the retrieval system to the LLM; design and iterate on prompts for different material types (word lists, sentences, slides); build initial output templates.
- [ ] Week 4: Develop the web-based chat interface; integrate the full pipeline end-to-end so a user can make a request and receive formatted output.
- [ ] Week 5: Systematic evaluation: test across a range of queries, check outputs against source data for accuracy, document failure modes (hallucinated words, incorrect grammar, retrieval misses).
- [ ] Week 6: Refine based on evaluation results; write up findings; prepare a presentation or poster for the SURP showcase.


## Repo Setup

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up dependencies

```bash
uv sync
```

### Run code

```bash
uv run python main.py
```