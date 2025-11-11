from fastapi import FastAPI
from entrypoints.events.github_webhook import router as github_webhook_router

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Agents backend is running!"}

app.include_router(github_webhook_router)
