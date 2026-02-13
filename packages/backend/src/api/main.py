from fastapi import FastAPI
from pydantic import BaseModel

from db import init_db

app = FastAPI(title="Agent Crossing API")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


class StatusResponse(BaseModel):
    status: str
    version: str


@app.get("/", response_model=StatusResponse)
async def get_status():
    return {"status": "online", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
