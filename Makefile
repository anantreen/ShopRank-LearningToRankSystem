.PHONY: setup test demo api

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -e '.[core,api]'

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 run_experiment.py

api:
	uvicorn api.main:app --reload

