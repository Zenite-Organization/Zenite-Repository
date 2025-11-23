from fastapi import FastAPI
from web.routes.github_webhook import router as github_webhook_router

import uvicorn

app = FastAPI()
app.include_router(github_webhook_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
