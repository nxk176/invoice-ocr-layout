.PHONY: install-dev format lint typecheck test smoke check

install-dev:
	python -m pip install -e ".[dev]"

format:
	python -m ruff format src tests scripts

lint:
	python -m ruff check src tests scripts

typecheck:
	python -m mypy src/invoice_ocr

test:
	python -m pytest

smoke:
	python -m pytest tests/integration/test_mock_pipeline.py

check: format lint typecheck test smoke

