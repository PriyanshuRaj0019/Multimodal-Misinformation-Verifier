
# Contributing to Multimodal Misinformation Verifier

Thank you for your interest in contributing to this project.

The Multimodal Misinformation Verifier is an evolving research and engineering project focused on evidence-grounded misinformation verification. Contributions that improve correctness, reproducibility, retrieval quality, testing, documentation, and engineering practices are welcome.

## 1. Getting Started

### Prerequisites

- Python 3.12
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/PriyanshuRaj0019/Multimodal-Misinformation-Verifier.git
cd Multimodal-Misinformation-Verifier
```

### Set up the environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 2. Development Standards

Please follow these principles:

- Write readable, maintainable Python code.
- Follow the existing project structure and conventions.
- Keep functions focused and avoid unnecessary abstractions.
- Handle invalid inputs and failure conditions explicitly.
- Avoid hardcoded machine-specific paths.
- Do not commit datasets, generated indexes, credentials, or local environment files.
- Document meaningful changes and experimental assumptions.
- Do not present experimental results as production-ready performance.

## 3. Testing and Code Quality

Before submitting changes, run the following commands:

```bash
python -m pytest -v
python -m ruff check scripts tests
python -m ruff format --check scripts tests
```

All tests and quality checks should pass.

For new functionality:

- Add tests for expected behavior.
- Include edge cases and invalid-input scenarios.
- Add regression tests when fixing bugs.
- Keep tests deterministic and independent of unavailable external datasets where practical.

## 4. Dataset and Experiment Guidelines

- Do not commit raw datasets or large generated evidence indexes.
- Do not include private data, credentials, or personally identifiable information.
- Keep generated experiment outputs separate from source code.
- Record dataset splits, evaluation methodology, and relevant configuration.
- Clearly distinguish retrieval metrics, classification metrics, and evidence-mapping coverage.
- Avoid claiming improvements without reproducible evaluation.

## 5. Submitting Changes

1. Check existing issues and pull requests before starting substantial work.
2. For significant changes, open an issue to discuss the proposed approach.
3. Keep changes focused and avoid unrelated modifications.
4. Add or update tests and documentation where necessary.
5. Run the required quality checks.
6. Submit a pull request against the `main` branch.

### Pull request descriptions should include

- What changed.
- Why the change is necessary.
- How the change was implemented.
- Tests and validation performed.
- Relevant issue references.
- Known limitations or follow-up work.

## 6. Reporting Bugs and Requesting Features

When reporting a bug, include:

- A clear description of the problem.
- Steps to reproduce it.
- Expected and actual behavior.
- Relevant error messages or logs.
- Python version and operating system.
- A minimal reproducible example, where possible.

For feature requests, explain the problem, proposed solution, expected benefit, and potential limitations.

## 7. Research and Technical Contributions

Contributions involving retrieval, machine learning, evidence ranking, or verification should include:

- A clear problem definition.
- Relevant baseline comparisons.
- Reproducible evaluation methodology.
- Appropriate metrics.
- Limitations and failure-case analysis.

Evidence-grounded verification and transparent uncertainty are important project principles.

## 8. Questions

For questions or proposed improvements, open a GitHub issue with sufficient technical context.

Thank you for helping improve this project.
