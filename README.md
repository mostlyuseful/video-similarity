# Video Similarity

A Python project for analyzing, comparing, and processing video files to identify similarities and differences. This document provides all necessary information for setup, usage, development, testing, contribution, and maintenance.

---

## Project Setup

### Prerequisites

- Python 3.10 or newer
- [pip](https://pip.pypa.io/en/stable/)
- [Ruff](https://docs.astral.sh/ruff/) (for linting)
- [pytest](https://docs.pytest.org/en/stable/) (for testing)
- Recommended: virtual environment (venv or conda)

### Installation Steps

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-org/video-similarity.git
   cd video-similarity
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install development tools:**
   ```sh
   pip install ruff pytest
   ```

### Initial Configuration

- Configuration files are located in [`src/video_similarity/config.py`](src/video_similarity/config.py:1).
- Adjust settings as needed for your environment.
- Templates are in [`src/video_similarity/templates/`](src/video_similarity/templates/).

---

## Usage Instructions

### Running the Project

- Main entry point: [`src/video_similarity/main.py`](src/video_similarity/main.py:1)
- To run the application:
  ```sh
  python src/video_similarity/main.py
  ```

### Key Commands

- **Run main processing:**
  ```sh
  python src/video_similarity/main.py --input <video_path>
  ```
- **View output:**
  Output files are generated in the working directory or as configured.

### Example Workflow

1. Place your video files in the desired directory.
2. Run the main script with appropriate arguments.
3. Review the output and logs for similarity results.

---

## Coding Standards

### Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Use 4 spaces for indentation.
- Limit lines to 88 characters.

### Naming Conventions

- Modules: `snake_case`
- Classes: `PascalCase`
- Functions & variables: `snake_case`
- Constants: `UPPER_CASE`

### Recommended Patterns

- Use type hints for all functions.
- Prefer explicit imports.
- Organize code into logical modules under [`src/video_similarity/`](src/video_similarity/).

### Ruff Usage

- Run Ruff to check code style and linting:
  ```sh
  ruff src/
  ```
- Fix issues automatically:
  ```sh
  ruff check src/ --fix
  ```
- Ruff configuration is managed in [`pyproject.toml`](pyproject.toml:1).

---

## Testing Guidelines

### Running Tests

- All tests are in [`tests/`](tests/) and follow the pytest structure.
- To run all tests:
  ```sh
  pytest
  ```

### Coverage Expectations

- Aim for 90%+ code coverage.
- Use pytest-cov for coverage reports:
  ```sh
  pytest --cov=src/video_similarity
  ```

### Test Organization

- Place unit tests in files named `test_*.py`.
- Use fixtures in [`tests/conftest.py`](tests/conftest.py:1).
- Separate integration and unit tests as needed.

---

## Contribution Protocols

### Branching Strategy

- Use `main` for stable releases.
- Create feature branches from `main`:
  ```
  git checkout -b feature/short-description
  ```

### Pull Request Process

1. Fork the repository and create your branch.
2. Commit changes with clear messages.
3. Ensure all tests pass and code is linted.
4. Open a pull request with a detailed description.

### Code Review Requirements

- All PRs require at least one approval.
- Address all review comments before merging.
- Ensure documentation is updated for new features.

---

## Maintenance Recommendations

### Update Procedures

- Regularly update dependencies:
  ```sh
  pip install --upgrade -r requirements.txt
  ```
- Review and update [`pyproject.toml`](pyproject.toml:1) as needed.

### Dependency Management

- Use `requirements.txt` for runtime dependencies.
- Use `pyproject.toml` for development and linting tools.
- Remove unused dependencies promptly.

### Documentation Upkeep

- Keep this README and [`spec.md`](spec.md:1) up to date.
- Document all public APIs and modules.
- Update usage examples with each release.

---
