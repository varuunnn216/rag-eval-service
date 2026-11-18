# Contributing

Thank you for contributing to `rag-eval-service`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

## Pull Request Guidelines

- Work on a feature/chore branch; do not commit directly to `main`.
- Never commit `.env` files, API keys, or local `.pgdata/` artifacts.
- Keep commits focused and verify unit tests pass before opening a PR.
