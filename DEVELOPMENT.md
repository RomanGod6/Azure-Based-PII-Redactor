# PII Redactor Pro Development Guide

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Poetry (installed automatically by setup.py)
- Git (optional, for version control)

### Quick Development Setup

```bash
# Complete development setup
make dev

# Or step by step:
python setup.py       # Install Poetry and dependencies
poetry install --with dev    # Include dev dependencies
```

## 📦 Project Structure

```
pii_redactor_pro/
├── pyproject.toml           # Poetry configuration & dependencies
├── Makefile                 # Cross-platform build commands
├── setup.py                 # Modern installation script
├── .env.template           # Environment template
├── .env                    # Your Azure credentials (auto-created)
├── requirements.txt        # Legacy pip requirements
├── pii_redactor_app.py     # Main GUI application
├── azure_pii_detector.py   # Azure AI integration
├── batch_process.py        # Batch processing script
└── sample_zendesk_data.csv # Sample test data
```

## 🔧 Development Commands

### Essential Commands

```bash
make help          # Show all available commands
make setup         # Complete project setup
make run           # Start the application
make dev           # Setup development environment
```

### Code Quality

```bash
make format        # Format code with black & isort
make lint          # Run flake8 & mypy linting
make check         # Run format + lint + test
```

### Testing

```bash
make test          # Run pytest tests
make clean         # Clean temporary files
```

### Dependencies

```bash
make deps-update   # Update all dependencies
make deps-show     # Show dependency tree
poetry add package # Add new dependency
poetry add --group dev package  # Add dev dependency
```

## 🎨 Code Style

We use modern Python tooling:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Auto-format your code:

```bash
make format
```

### Check code quality:

```bash
make lint
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test
poetry run pytest tests/test_specific.py

# Run with coverage
poetry run pytest --cov=pii_redactor_pro
```

## 📋 Adding Dependencies

### Runtime Dependencies

```bash
poetry add package-name
```

### Development Dependencies

```bash
poetry add --group dev package-name
```

### Update pyproject.toml

Dependencies are automatically added to `pyproject.toml`. The old `requirements.txt` is maintained for compatibility.

## 🚀 Building & Distribution

```bash
# Build distribution packages
make build

# Clean build artifacts
make clean
```

## 🔍 Development Workflow

1. **Setup**: `make dev`
2. **Code**: Make your changes
3. **Format**: `make format`
4. **Check**: `make check`
5. **Test**: `make test`
6. **Run**: `make run`

## 🌍 Poetry Virtual Environment

Poetry automatically creates and manages virtual environments:

```bash
# Activate shell in virtual environment
poetry shell

# Run commands in virtual environment
poetry run python script.py

# Show virtual environment info
poetry env info
```

## 📝 Configuration Files

### pyproject.toml

Modern Python project configuration:

- Dependencies
- Build settings
- Tool configurations (black, isort, mypy)

### Makefile

Cross-platform build commands that work on Windows, macOS, and Linux.

### .env / .env.template

Environment variables for Azure credentials.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `make check`
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create pull request

## 📚 Learning Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Make Tutorial](https://makefiletutorial.com/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Azure Text Analytics](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/)
