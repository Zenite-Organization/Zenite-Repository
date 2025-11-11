# from fastapi import FastAPI
# from entrypoints.events.github_webhook import router as github_webhook_router

# app = FastAPI()

# @app.get("/")
# def root():
#     return {"message": "AI Agents backend is running!"}

# app.include_router(github_webhook_router)
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API funcionando!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

