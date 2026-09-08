# Contributing to AgentAblit

Thank you for your interest in contributing! AgentAblit is a security-research project, and we
welcome contributions that improve the codebase, documentation, test coverage, and usability.

## Getting Started

1. **Fork** the repository and clone your fork
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Copy the config template:**
   ```sh
   cp config.example.yaml config.yaml
   ```
4. **Run the tests:**
   ```sh
   python -m pytest tests/ -v
   ```

## Development Setup

The project uses a simple Python structure under `src/`. No build step is required — the code
runs directly.

```sh
# Run the relay (requires a host model API configured in config.yaml)
cd src && python -m proxy.message_forward

# Run the config panel
uvicorn panel.server:app --port 8790

# Run tests
python -m pytest tests/ -v
```

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include steps to reproduce, expected behavior, and actual behavior
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)

### Code Contributions

1. **Create a branch** from `main` for your change
2. **Write tests** for new functionality
3. **Follow existing code style** — the project uses type hints, docstrings, and
   consistent naming conventions
4. **Keep commits focused** — one logical change per commit
5. **Open a pull request** with a clear description of what and why

### What We're Looking For

- **Test coverage** — expanding the test suite is the highest-priority contribution
- **Documentation** — examples, integration guides, architecture docs
- **Bug fixes** — especially in the controller logic or edge cases
- **New context resolution strategies** — see `src/strategies/reconstruct/`
- **Integration examples** — demonstrate AgentAblit with popular agent frameworks

### Code Style

- Python 3.10+ (type hints with `X | Y` union syntax)
- Docstrings on public functions and classes
- Constants in `UPPER_SNAKE_CASE`
- Environment variables prefixed with `ABLIT_`
- Tests use `pytest` style (functions, not classes)

## Code of Conduct

Be respectful and constructive. We're all here to make LLM agents more reliable.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache-2.0 License](LICENSE).
