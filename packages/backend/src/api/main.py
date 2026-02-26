from pathlib import Path

from agents.seed_loader import SeedLoader
from fastapi import FastAPI
from pydantic import BaseModel

from db import init_db

app = FastAPI(title="Agent Crossing API")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    seed_dir = Path(__file__).resolve().parents[2] / "persona"
    app.state.seed_loader = SeedLoader(seed_dir)
    app.state.agent_seeds = app.state.seed_loader.load_all()


class StatusResponse(BaseModel):
    status: str
    version: str


@app.get("/", response_model=StatusResponse)
async def get_status():
    return {"status": "online", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
