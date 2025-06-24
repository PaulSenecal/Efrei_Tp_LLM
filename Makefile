# Makefile for Culinary Assistant LLM

PYTHON = python3
PIP = pip3
PROJECT_NAME = culinary-assistant-llm
PYTHON_FILES = $(shell find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*")
TEST_FILES = $(shell find tests -name "*.py" 2>/dev/null || echo "")

GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
BLUE = \033[0;34m
NC = \033[0m

.PHONY: help install train demo chat test data clean lint format

help:
	@echo "Culinary Assistant LLM - Makefile"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Train the model"
	@echo "  make demo        - Quick demonstration"
	@echo "  make chat        - Interactive interface"
	@echo "  make test        - Automatic tests"
	@echo "  make data        - Show data sources information"
	@echo "  make clean       - Clean temporary files"
	@echo "  make lint        - Check code style"
	@echo "  make format      - Format code"
	@echo "  make run         - Launch application (alias for chat)"
	@echo ""

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully!"

train:
	@echo "Training model..."
	$(PYTHON) main.py train
	@echo "Training completed!"

demo:
	@echo "Quick demonstration..."
	$(PYTHON) main.py demo
	@echo "Demonstration completed!"

chat:
	@echo "Launching interactive chat..."
	$(PYTHON) main.py chat

run: chat

test:
	@echo "Running tests..."
	$(PYTHON) main.py test
	@echo "Tests completed!"

data:
	@echo "Showing data sources information..."
	$(PYTHON) main.py data
	@echo "Data information displayed!"

clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log" -delete
	@find . -type f -name ".DS_Store" -delete
	@echo "Cleaning completed!"

lint:
	@echo "Checking code style..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 $(PYTHON_FILES) --max-line-length=100 --ignore=E203,W503; \
		echo "Style check completed!"; \
	else \
		echo "flake8 not installed. Install with: pip install flake8"; \
	fi

format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black $(PYTHON_FILES) --line-length=100; \
		echo "Formatting completed!"; \
	else \
		echo "black not installed. Install with: pip install black"; \
	fi