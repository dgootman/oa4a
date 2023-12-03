"""OA4A Server"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """Print Hello World"""
    return {"Hello": "World"}
