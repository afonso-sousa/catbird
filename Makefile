PACKAGE=catbird
UNIT_TESTS=tests

all: static-checks coverage

.PHONY: all

style:
	$(info Running style analysis...)
	poetry run black $(PACKAGE)

typecheck:
	$(info Running static type analysis...)
	poetry run mypy $(PACKAGE)

doccheck:
	$(info Running documentation analysis...)
	poetry run pydocstyle -v $(PACKAGE)
	
spellcheck:
	$(info Running spell checker...)
	poetry run codespell $(PACKAGE)

static-checks: style spellcheck doccheck

unit-tests:
	$(info Running unit tests...)
	poetry run pytest -v $(UNIT_TESTS)

coverage:
	$(info Running coverage analysis with JUnit xml export...)
	poetry run pytest --cov=./ --cov-report=xml

