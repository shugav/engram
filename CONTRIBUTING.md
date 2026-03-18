# Contributing to Engram

First off, thank you for considering contributing to Engram.

**First-time contributors are very welcome.** If this is your first open-source contribution, you are in the right place.

## Community Expectations

This project is committed to a respectful, constructive, low-drama collaboration style.

- Be kind, direct, and specific.
- Assume good intent.
- Critique code, not people.

Please read the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to follow it.

## Quick Path (First Contribution)

1. Open or pick an issue to work on.
2. Leave a short comment so work is not duplicated.
3. Fork the repo and create a branch.
4. Make a focused change.
5. Open a pull request using the template.

Small docs improvements are a great first PR.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/shugav/engram.git
cd engram
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment:

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Running Locally

```bash
python -m engram                                # stdio mode (default)
python -m engram --transport sse                # SSE mode
python -m engram --transport sse --port 9000    # SSE custom port
```

## Project Structure

```text
src/engram/
├── server.py      # MCP tools and server entry point
├── db.py          # SQLite schema and data access
├── search.py      # Three-layer search engine (BM25 + vector + graph)
├── embeddings.py  # OpenAI embedding client
├── chunker.py     # Text chunking with overlap
└── types.py       # Pydantic data models
```

## Ways to Contribute

- **Bug reports**: include clear steps to reproduce and expected behavior.
- **Feature requests**: include use case, motivation, and alternatives.
- **Code contributions**: keep changes scoped and easy to review.
- **Documentation**: improve clarity, examples, and onboarding.

## Pull Request Guidelines

- Keep each PR focused on one logical change.
- Link the related issue when possible.
- Update docs when behavior changes.
- Add or update tests when practical.
- Use clear commit messages and PR descriptions.

## Review Process

Maintainers aim to provide feedback that is:

- Respectful and actionable
- Focused on correctness and maintainability
- Supportive for new contributors

If feedback is unclear, ask follow-up questions. That is encouraged.

## Need Help?

- Open an issue using the templates.
- Include context, logs, and what you already tried.
- If you are unsure where to start, mention that you want a "good first issue."

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
