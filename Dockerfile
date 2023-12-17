# Based on https://fastapi.tiangolo.com/deployment/docker/#docker-image-with-poetry

FROM python:3 as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


FROM python:3

WORKDIR /app

COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./oa4a /app/oa4a

EXPOSE 8000

CMD ["uvicorn", "oa4a.server:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]