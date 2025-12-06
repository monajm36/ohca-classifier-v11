# Contributing to OHCA Classifier

Thank you for your interest in contributing to the OHCA Classifier project! 

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [Issues](https://github.com/monajm36/ohca-classifier/issues)
2. If not, create a new issue with:
   - Clear title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Python version and dependencies

### Suggesting Enhancements

We welcome ideas for improvements:

- New features for temporal/location detection
- Additional model architectures
- Performance optimizations
- Better documentation
- New use cases

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/monajm36/ohca-classifier.git
   cd ohca-classifier
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: new temporal feature for arrest timing"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone repository
git clone https://github.com/monajm36/ohca-classifier.git
cd ohca-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Keep functions focused and small

Example:
```python
def extract_location_features(text: str) -> dict:
    """
    Extract location features from clinical text.
    
    Args:
        text: Clinical note text
        
    Returns:
        Dictionary with ohca_count, ihca_count, location_score
    """
    # Implementation
    pass
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=ohca_classifier tests/

# Run specific test
pytest tests/test_features.py::test_location_extraction
```

## Areas for Contribution

### High Priority
- [ ] Multi-institution validation
- [ ] Explainability features (LIME, SHAP)
- [ ] Real-time inference optimization
- [ ] Additional temporal features

### Medium Priority
- [ ] Web demo (Gradio/Streamlit)
- [ ] API endpoint deployment
- [ ] Non-English language support
- [ ] Additional base models (Clinical-BERT, etc.)

### Documentation
- [ ] Tutorial notebooks
- [ ] Video walkthrough
- [ ] Blog posts
- [ ] Use case examples

## Questions?

Feel free to:
- Open an issue
- Email: [your-email]
- Discussion forum: [if applicable]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for helping improve cardiac arrest research! ❤️
