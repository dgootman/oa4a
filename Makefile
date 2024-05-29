.PHONY: release build install model format lint test dev

SHELL := /bin/bash -euo pipefail

release: build test

build: install model format lint

install:
	poetry install

define MODEL_HEADER
"""
OpenAI Data Models

Auto-generated from https://github.com/openai/openai-openapi
"""

# pylint: skip-file
endef
export MODEL_HEADER

model: oa4a/model.py

# Generate a Pydantic model for OpenAI
# Using https://github.com/koxudaxi/datamodel-code-generator
oa4a/model.py: openai-openapi/openapi.yaml openapi.jq
	source $$(poetry env info --path)/bin/activate && \
	cat openai-openapi/openapi.yaml | \
		yq -Y -f openapi.jq | \
		datamodel-codegen --input-file-type openapi \
			--custom-file-header "$${MODEL_HEADER}" \
			--use-schema-description --use-field-description \
			--disable-timestamp --allow-extra-fields --enum-field-as-literal all \
			--output oa4a/model.py --output-model-type pydantic_v2.BaseModel

format:
	poetry run isort oa4a
	poetry run black oa4a

lint:
	poetry run pylint oa4a

test:
	poetry run pytest

dev:
	poetry run fastapi dev oa4a/server.py