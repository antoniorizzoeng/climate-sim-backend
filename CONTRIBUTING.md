# Contributing Guidelines

Thank you for considering contributing to climate-sim-backend!

---

## Development Workflow

1. Fork the repository and create your branch:
   git checkout -b feature/your-feature

2. Set up the environment:
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   pip install pre-commit
   pre-commit install

3. Run tests to ensure everything passes:
   pytest -q

4. Run formatting and linting checks:
   black --check .
   isort --check-only .
   flake8

5. Submit a Pull Request with a clear description of what was changed and why.

---

## Reporting Issues

Use the GitHub Issues page and include:
- Reproduction steps (e.g., API call examples)
- Expected vs actual behavior
- Logs or traceback output if relevant
- Your environment info (python --version, OS, Docker version)

---

## Code Style

- Follows PEP 8 conventions.
- Formatting enforced via:
  - Black (black .)
  - isort (isort .)
  - flake8 (for linting)
- Hooks are automatically run via pre-commit on each commit.

---

## Testing & Coverage

All code changes must include unit or integration tests.

Run the test suite:
   pytest --cov=app --cov-report=term-missing

Coverage should remain ≥ 90%.
Pull Requests that reduce coverage below this threshold may be requested to add additional tests.

To generate an HTML coverage report:
   pytest --cov=app --cov-report=html
   open htmlcov/index.html

---

## Docker Development

To test the app in Docker:
   docker compose up --build

For the real MPI and ffmpeg integration tests, follow the local setup in README.md and run `pytest -q`.

---

## Security & Best Practices

- Do not commit secrets — use .env.example as a template.
- Dependencies are managed via requirements.txt and updated weekly by Dependabot.
- Code is scanned by CodeQL for vulnerabilities.

---

## Pull Request Checklist

- [ ] Code is formatted (black .)
- [ ] Imports sorted (isort .)
- [ ] Linting passes (flake8)
- [ ] All tests pass locally (pytest)
- [ ] New code includes tests and docstrings
- [ ] Coverage ≥ 90%

---

Happy hacking! 
Your contributions help make climate simulation more accessible and reproducible.
