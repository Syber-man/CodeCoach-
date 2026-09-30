# CodeCoach - Agentic AI assistant for competitive programming

Gen AI Capstone 2026. Answers only from 12 verified algorithm notes (RAG) and calls live tools
(Codeforces rating, complexity checker, date/time). Every answer is scored for faithfulness and
retried automatically when the score is below 0.7.

**Stack:** LangGraph, ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2), Groq (llama-3.3-70b-versatile), Streamlit.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # then put your free key from console.groq.com in .env
streamlit run app.py        # chat UI
python eval.py              # runs the 10 tests, writes results.csv
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo.
2. On share.streamlit.io choose the repo, main file `app.py`.
3. In Advanced settings > Secrets add: `GROQ_API_KEY = "your_key"`.

## Files
| File | Purpose |
|---|---|
| agent.py | LangGraph pipeline (10 nodes) |
| knowledge_base.py | 12 topic notes |
| tools.py | Codeforces, complexity, datetime tools |
| app.py | Streamlit chat interface |
| eval.py | Test suite and metrics |
