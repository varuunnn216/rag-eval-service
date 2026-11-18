# Contributing

Thanks for checking out **RAG Workbench**!

## Local Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-local.txt
   ```
2. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` and `DATABASE_URL`.
3. Seed the sample corpus and start the server:
   ```bash
   python3 -m app.ingestion.seed
   uvicorn app.main:app --reload --port 8000
   ```

## Running Tests

```bash
# Unit tests (no external server required)
python3 -m unittest discover -s tests -v

# Live HTTP regression and security checks against a running server
BASE_URL=http://127.0.0.1:8000 python3 tests/regression.py
BASE_URL=http://127.0.0.1:8000 python3 -m unittest tests/test_security.py -v
```
