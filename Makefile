###############################################################################
# COMMANDS
###############################################################################
.PHONY: clean
## Clean python cache file.
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name .coverage -delete
	find . -name '.coverage.*' -delete
	find . -name 'codeclimate.*' -delete
	find . -name 'requirements*.txt' -delete
	find . -name 'report.html' -delete
	find . -name cov.xml -delete
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +

.PHONY: install-pdm
## install pdm before environment setup
install-pdm:
	python -m pip install -U \
	    pip setuptools wheel pdm

.PHONY: update-lock
## update pdm.lock
update-lock:
	pdm update --no-sync

.PHONY: deploy-dev
## deploy dev environment
deploy-dev:
	pdm sync -G dev -G repl --clean

.PHONY: format
## isort and yapf formatting
format:
	pdm run isort src tests
	pdm run yapf -i -r src tests

.PHONY: lint
## pylint check
lint:
	pdm run pylint --rcfile=.pylintrc \
	    --exit-zero \
	    --msg-template='{path}:{line}:{column}:**[{msg_id}]** ({category}, {symbol})<br>{msg}' \
	    --output-format=parseable src tests

.PHONY: lint-type
## type checking linter with mypy
lint-type:
	pdm run mypy

.PHONY: test
test:
	PYTHONPATH=./src \
	    pdm run pytest -s -v --cov=app --cov-config=pyproject.toml \
	    > coverage.txt
