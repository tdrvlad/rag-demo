from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
from index import index_module  # Ensure the Index class is correctly imported from where you have defined it

app = FastAPI()


class TextData(BaseModel):
    text: str


@app.post("/add/")
async def add_sample(text_data: TextData):
    """
    Endpoint to add text to the index.
    """
    try:
        index_module.add_text(text_data.text)
        return {"message": "Text added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/")
async def search(query: TextData):
    """
    Endpoint to perform a search in the index.
    """
    try:
        results = index_module.search(query.text)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
