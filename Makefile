.PHONY: install run

install:
	poetry install

run:
	poetry run uvicorn oa4a.server:app --reload