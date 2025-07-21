# Contributing to MedAI-Xpert

We welcome contributions to MedAI-Xpert! Your insights and improvements are valuable. Please follow these guidelines to ensure a smooth contribution process.

## Table of Contents

* [How to Contribute](#how-to-contribute)
* [Code of Conduct](#code-of-conduct)
* [Setting Up Your Development Environment](#setting-up-your-development-environment)
* [Coding Style and Standards](#coding-style-and-standards)
* [Commit Messages](#commit-messages)
* [Pull Request Guidelines](#pull-request-guidelines)
* [Testing](#testing)
* [Reporting Bugs](#reporting-bugs)
* [Feature Requests](#feature-requests)

## How to Contribute

1.  **Fork** the repository on GitHub.
2.  **Clone** your forked repository locally.
3.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/issue-description`.
4.  **Make your changes**, ensuring they adhere to the coding standards.
5.  **Write unit and integration tests** for your changes.
6.  **Run all tests** to ensure no regressions are introduced.
7.  **Write clear, atomic commit messages** following the Conventional Commits specification.
8.  **Push your branch** to your forked repository.
9.  **Open a Pull Request** against the `develop` branch of the main `MedAI-Xpert` repository.

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md - create this optionally if desired). We are committed to fostering an open and welcoming environment.

## Setting Up Your Development Environment

Refer to the [README.md](README.md) for detailed instructions on setting up the local development environment using Docker Compose.

## Coding Style and Standards

* **Python:** Adhere to PEP 8.
* **Formatting:** We use `black` for code formatting. Run `black src/` before committing.
* **Linting:** We use `flake8`. Ensure your code passes `flake8 src/` checks.
* **Imports:** We use `isort` for sorting imports. Run `isort src/` before committing.
* **Type Hinting:** Use Python's type hints extensively for clarity and maintainability.
* **Docstrings:** Write clear and concise docstrings for all functions, classes, and modules.

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in generating changelogs and understanding the project history.

**Format:** `<type>(<scope>): <short description>`

* **type:** `feat` (new feature), `fix` (bug fix), `docs` (documentation), `style` (formatting, no code change), `refactor` (code refactoring), `test` (adding tests), `chore` (maintenance, build tools etc.)
* **scope (optional):** `api`, `models`, `data_ingestion`, `docs`, `ci`, etc.
* **Examples:**
    * `feat(api): Add patient prediction endpoint`
    * `fix(models): Correct bug in image detector preprocessing`
    * `docs(readme): Update installation instructions`
    * `test(data_ingestion): Add unit tests for DICOM extractor`

## Pull Request Guidelines

* Ensure your branch is up-to-date with the `develop` branch.
* Provide a clear and concise description of your changes in the PR.
* Reference any related issues (`Fixes #123`, `Closes #456`).
* Ensure all CI/CD checks pass.
* Request reviews from relevant team members.

## Testing

* **Unit Tests:** Located in `src/tests/unit/`. Focus on individual functions or classes.
* **Integration Tests:** Located in `src/tests/integration/`. Focus on interactions between components or services.
* Run tests using `pytest src/tests/`.

## Reporting Bugs

If you find a bug, please open a new issue on the GitHub repository. Provide:
* A clear and concise description of the bug.
* Steps to reproduce the behavior.
* Expected vs. actual behavior.
* Any relevant logs or screenshots.
* Your environment details (OS, Python version, dependencies).

## Feature Requests

If you have an idea for a new feature, please open a new issue. Describe:
* The proposed feature.
* The problem it solves or the benefit it provides.
* Any potential challenges or alternatives.