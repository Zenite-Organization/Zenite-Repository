from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

# Ensure `src` folder (this file's dir) is on sys.path so `ai.*` imports work
sys.path.insert(0, os.path.dirname(__file__))

from ai.workflows.estimation_graph import run_estimation_flow

app = FastAPI()


class EstimateRequest(BaseModel):
    issue_description: str


@app.get("/")
def home():
    return {"message": "API funcionando!"}


@app.post("/estimate")
def estimate(req: EstimateRequest):
    try:
        result = run_estimation_flow(req.issue_description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

