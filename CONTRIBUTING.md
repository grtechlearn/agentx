# Contributing to AgentX

Thank you for your interest in contributing to AgentX! This guide will help you get started.

## Development Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/agentx.git
   cd agentx
   ```

2. **Install Python 3.11+**

   AgentX requires Python 3.11 or later. We recommend using [pyenv](https://github.com/pyenv/pyenv) to manage Python versions:

   ```bash
   pyenv install 3.11
   pyenv local 3.11
   ```

3. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

5. **Set up pre-commit hooks**

   ```bash
   pre-commit install
   ```

## Running Tests

We use **pytest** as our test framework.

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_agents.py

# Run tests matching a keyword
pytest -k "test_orchestrator"

# Run with coverage report
pytest --cov=agentx --cov-report=term-missing
```

## Code Style

- **Python 3.11+** — use modern syntax including `match` statements, `StrEnum`, and `type` aliases where appropriate.
- **Type hints** — all public functions and methods must have complete type annotations.
- **Pydantic** — use Pydantic `BaseModel` for data validation and settings. Prefer `model_validator` over custom `__init__`.
- **Async-first** — prefer `async def` for I/O-bound operations. Use `asyncio.TaskGroup` for concurrent work.
- **Formatting** — we use `ruff` for linting and formatting. Run before committing:

  ```bash
  ruff check .
  ruff format .
  ```

- **Docstrings** — use Google-style docstrings for modules, classes, and public functions.
- **Imports** — group as: stdlib, third-party, local. Let `ruff` handle sorting.

## Pull Request Process

1. **Create a branch** from `main` with a descriptive name:

   ```bash
   git checkout -b feat/add-new-tool
   ```

   Use prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`.

2. **Make your changes** in small, focused commits with clear messages.

3. **Ensure all checks pass** before pushing:

   ```bash
   ruff check .
   ruff format --check .
   pytest
   ```

4. **Open a pull request** against `main`:
   - Provide a clear title and description of what changed and why.
   - Reference any related issues (e.g., `Closes #42`).
   - Include test coverage for new functionality.
   - Update documentation if behaviour changes.

5. **Code review** — at least one maintainer approval is required before merging. Address review feedback with additional commits (do not force-push during review).

6. **Merge** — maintainers will squash-merge once approved.

## Issue Templates

### Bug Report

When filing a bug, please include:

- **Description** — a clear summary of the problem.
- **Steps to reproduce** — minimal code or commands to trigger the issue.
- **Expected behaviour** — what you expected to happen.
- **Actual behaviour** — what actually happened, including error messages or tracebacks.
- **Environment** — Python version, OS, and AgentX version (`agentx --version`).

### Feature Request

When proposing a feature, please include:

- **Problem statement** — what pain point does this address?
- **Proposed solution** — how you envision it working.
- **Alternatives considered** — other approaches you evaluated.
- **Additional context** — mockups, references, or related issues.

## Code of Conduct

By participating in this project you agree to treat all contributors with respect and follow professional standards of conduct. Be kind, be constructive, and assume good intent.
