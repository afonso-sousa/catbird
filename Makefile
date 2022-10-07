PACKAGE?=catbird
UNIT_TESTS=tests

all: static-checks coverage

.PHONY: all

formatting:
	$(info Formatting...)
	poetry run black $(PACKAGE)
sort_imports:
	$(info Sorting imports...)
	poetry run isort $(PACKAGE)

linting:
	$(info Getting linting suggestions...)
	poetry run flake8 $(PACKAGE)

typecheck:
	$(info Running static type analysis...)
	poetry run mypy $(PACKAGE)

doccheck:
	$(info Checking documentation...)
	poetry run pydocstyle -v $(PACKAGE)
	
spellcheck:
	$(info Spell checking...)
	poetry run codespell $(PACKAGE)

deadcode:
	$(info Searching for dead code...)
	poetry run vulture $(PACKAGE)

static-checks: formatting sort_imports linting spellcheck

unit-tests:
	$(info Running unit tests...)
	poetry run pytest -v $(UNIT_TESTS)

coverage:
	$(info Running coverage analysis with JUnit xml export...)
	poetry run pytest --cov=./ --cov-report=xml
