# Agent Crossing

Autonomous social simulation inspired by **Generative Agents (Park et al., 2023)**.
Built with **React 19 + Phaser 3** (frontend) and **FastAPI** (backend).

## Status

> Work in progress. Core architecture is set, cognitive loop is under active implementation.

## What this project aims to do

- NPCs plan their day, remember experiences, and react to unexpected events.
- Memories are retrieved by recency/importance/relevance scoring.
- Reflection generates higher-level insights from recent experiences.
- Multiple agents interact, exchange information, and form social dynamics.

## Tech Stack

- Frontend: React 19, Phaser 3, Zustand, Vite
- Backend: FastAPI, Pydantic, uvicorn, uv
- AI/Memory: local LLM (MLX on Apple Silicon), PostgreSQL + pgvector, sentence-transformers
- Monorepo: pnpm workspace

## Project Structure

```text
packages/
  shared/      # shared types/constants
  frontend/    # React + Phaser client
  backend/     # FastAPI + agent brain
```

## Getting Started

### Prerequisites

- Node.js 20+
- pnpm 9+
- Python 3.11+
- uv

### Install

```bash
pnpm install
uv sync --project packages/backend
```

### Run (dev)

```bash
# terminal 1
pnpm dev:backend

# terminal 2
pnpm dev:frontend
```

### PostgreSQL + pgvector (Docker)

```bash
docker compose up -d
```

Quick check:

```bash
docker compose exec postgres psql -U agent -d agent_crossing -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

If the result includes `vector`, pgvector is enabled.

### Tests

```bash
pnpm test:backend
# or
uv run --project packages/backend pytest -c packages/backend/pyproject.toml packages/backend/tests
```

## Roadmap

- See `TODO.md` for implementation checklist aligned with the paper.
- See `SPEC.md` for architecture and technical specification.

## Reference

- Paper: [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

## License

TBD
