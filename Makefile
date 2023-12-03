.PHONY: install run model

SHELL := /bin/bash -euo pipefail

install:
	poetry install

run:
	poetry run uvicorn oa4a.server:app --reload

define MODEL_HEADER
'''
OpenAI Data Models

Auto-generated from https://github.com/openai/openai-openapi
'''
endef
export MODEL_HEADER

model: oa4a/model.py

oa4a/model.py: openai-openapi/openapi.yaml
	source $$(poetry env info --path)/bin/activate && \
	cat openai-openapi/openapi.yaml | \
		yq -Y -f openapi.jq | \
		datamodel-codegen --input-file-type openapi \
			--custom-file-header "$${MODEL_HEADER}" \
			--use-schema-description --disable-timestamp --allow-extra-fields \
			--output oa4a/model.py --output-model-type pydantic_v2.BaseModel