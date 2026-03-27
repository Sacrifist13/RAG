PYTHON_CMD  := uv run python -m src index

BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m

.PHONY: all install run debug clean lint

all: install run

install:
	@echo "$(BOLD)🚀 Initializing project environment and syncing dependencies...$(RESET)"
	@pip install uv
	@uv sync
	@echo "\n$(BOLD)$(GREEN)✅ Environment setup complete.$(RESET)"

run:
	@echo "$(BOLD)🕹️  Executing main script...$(RESET)"
	$(PYTHON_CMD)

debug:
	@echo "$(BOLD)⚙️ Executing main script with pdb...$(RESET)"
	$(PYTHON_CMD) -m pdb

clean:
	@echo "$(BOLD)🗑️  Cleaning up cache...$(RESET)"
	rm -rf .mypy_cache __pycache__ src/__pycache__
	@echo "$(BOLD)$(GREEN)🧹 Cache is clean.$(RESET)"

lint:
	@echo "$(BOLD)🔎 Running static code analysis...$(RESET)"
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=.venv
