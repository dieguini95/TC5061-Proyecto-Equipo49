#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo49
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
QUALITY_PYTHON = .venv-quality/bin/python
SERVING_PYTHON ?= $(PYTHON_INTERPRETER)
QUESTION ?= p2
ENGINE ?= docker
PORT ?= 8000

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests

## Run DeepChecks data validation (needs .venv-quality, see requirements-quality.txt)
.PHONY: data-validation
data-validation:
	$(QUALITY_PYTHON) -m mibici_ml.data_validation

## Run PyTest for DeepChecks data validation (needs .venv-quality)
.PHONY: data-validation-tests
data-validation-tests:
	$(QUALITY_PYTHON) -m pytest -q tests/test_data_validation.py

## Run P1 data validation without changing the P2/P3 reports
.PHONY: data-validation-p1
data-validation-p1:
	$(QUALITY_PYTHON) -m mibici_ml.data_validation_p1

## Run P1 validation tests in the isolated quality environment
.PHONY: data-validation-p1-tests
data-validation-p1-tests:
	$(QUALITY_PYTHON) -m pytest -q tests/test_data_validation_p1.py

## Run offline repository security checks
.PHONY: security security-audit
security:
	bash scripts/run_security_checks.sh

## Run offline checks and dependency audit
security-audit:
	bash scripts/run_security_checks.sh --with-audit


## Exporta el champion del Registry a models/serving/<QUESTION> (QUESTION=p1|p2|p3)
.PHONY: serving-export
serving-export:
	$(SERVING_PYTHON) -m mibici_ml.serving.export --question $(QUESTION)

## Construye la imagen del servicio con ENGINE=docker|podman y QUESTION=p2
.PHONY: serving-build
serving-build:
	$(ENGINE) build -f Containerfile --build-arg QUESTION=$(QUESTION) -t mibici-$(QUESTION):latest .

## Ejecuta el contenedor del servicio en el puerto PORT (8000 por defecto)
.PHONY: serving-run
serving-run:
	$(ENGINE) run --rm -p $(PORT):8000 --name mibici-$(QUESTION) mibici-$(QUESTION):latest

## Pruebas del servicio de inferencia (no requieren contenedor)
.PHONY: serving-test
serving-test:
	$(SERVING_PYTHON) -m pytest -q tests/test_serving_*.py


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mibici_ml/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
