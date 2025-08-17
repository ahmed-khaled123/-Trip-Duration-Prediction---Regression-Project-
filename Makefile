.PHONY: setup eda format clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt || true
	@echo "✓ Environment ready"

eda:
	@echo "Opening Jupyter... (or open the notebook in VS Code)"
	jupyter lab

format:
	@echo "Add your formatter here if needed"

clean:
	rm -rf __pycache__ .pytest_cache .ipynb_checkpoints
