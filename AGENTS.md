# Agent Crossing Development Guide

**Tech Stack**: TypeScript, React 19, Tailwind CSS 4, Zustand, WorkAdventure (Phaser 3), FastAPI, Transformers  
**Status**: Greenfield project (specification phase)

---

## Build, Lint, and Test Commands

### Frontend (TypeScript/React)
```bash
# Development
npm install                 # Install dependencies
npm run dev                 # Start dev server
npm run build               # Production build

# Code Quality
npm run type-check          # TypeScript type checking
npm run lint                # ESLint
npm run lint:fix            # Auto-fix lint issues
npm run format              # Prettier formatting
npm run format:check        # Check formatting

# Testing
npm test                    # Run all tests
npm test -- --watch         # Watch mode
npm test -- path/to/test    # Run single test file
npm run test:coverage       # Coverage report
```

### Backend (Python/FastAPI)
```bash
# Development
pip install -r requirements.txt     # Install dependencies (or: uv pip install -r requirements.txt)
uvicorn main:app --reload           # Start dev server

# Code Quality
mypy .                      # Type checking
ruff check .                # Linting
ruff format .               # Formatting

# Testing
pytest                          # Run all tests
pytest tests/test_agents.py     # Run single test file
pytest -k test_memory           # Run tests matching pattern
pytest --cov                    # Coverage report
```

---

## Code Style Guidelines

### Import Organization
Order with blank lines between groups:
```typescript
// 1. External dependencies
import React, { useState } from 'react';
import { create } from 'zustand';

// 2. Internal absolute imports (@/)
import { AgentService } from '@/services/agent.service';
import { useAgentStore } from '@/stores/agent.store';

// 3. Relative imports
import { AgentCard } from './AgentCard';

// 4. Type imports
import type { Agent, AgentMemory } from '@/types/agent';
```

**Rules**: Use `@/` for cross-module imports, `./` only within same module, alphabetize within groups.

### Naming Conventions

**Files**:
- Components: `AgentCard.tsx` (PascalCase)
- Services: `agent.service.ts` (camelCase + .service)
- Stores: `agent.store.ts` (camelCase + .store)
- Hooks: `useAgent.ts` (camelCase, use prefix)
- Types: `agent.types.ts` (camelCase + .types)
- Utils: `vector.util.ts` (camelCase + .util)
- Tests: `agent.service.test.ts` (same as source + .test)

**Code**:
```typescript
// Functions: camelCase, verb-noun pattern
function calculateAgentAffinity(a: Agent, b: Agent): number {}

// Variables: camelCase
const agentList: Agent[] = [];

// Constants: UPPER_SNAKE_CASE
const MAX_MEMORY_TOKENS = 2048;

// Classes/Interfaces/Types: PascalCase
class AgentBrain {}
interface Agent {}
type AgentId = string;

// Private: prefix with underscore
private _cache: Map<string, Agent>;
```

### TypeScript Patterns

**Always use explicit types**:
```typescript
// ✅ Good
function processMemory(memory: AgentMemory): void { }

// ❌ Bad - implicit any
function processMemory(memory) { }
```

**Strict null checks**:
```typescript
function getAgent(id: string): Agent | null {
  return agentMap.get(id) ?? null;
}

// Use optional chaining and nullish coalescing
const name = agent?.profile?.name ?? 'Unknown';
```

**NEVER use `any`, `as any`, `@ts-ignore`**:
```typescript
// ❌ NEVER do this
const data = response as any;
// @ts-ignore
const value = obj.unknownProp;

// ✅ Define proper types
interface Response { data: AgentData; }
const data: Response = response;
```

### Error Handling

**Result type pattern** (recommended):
```typescript
type Result<T, E = Error> = 
  | { ok: true; value: T }
  | { ok: false; error: E };

async function fetchAgent(id: string): Promise<Result<Agent>> {
  try {
    const data = await api.get(`/agents/${id}`);
    return { ok: true, value: data };
  } catch (error) {
    return { ok: false, error: error as Error };
  }
}

// Usage
const result = await fetchAgent('agent-1');
if (result.ok) {
  console.log(result.value.name);
} else {
  console.error(result.error.message);
}
```

### File Organization

```
src/
├── components/       # React components (UI only, no business logic)
│   ├── Agent/
│   └── Memory/
├── services/         # Business logic & API calls
│   ├── agent.service.ts
│   └── memory.service.ts
├── stores/           # Zustand state management
│   ├── agent.store.ts
│   └── ui.store.ts
├── hooks/            # Custom React hooks (reusable stateful logic)
│   └── useAgent.ts
├── types/            # TypeScript type definitions
│   └── agent.types.ts
├── utils/            # Pure utility functions (no side effects)
│   └── vector.util.ts
└── config/           # Constants & configuration
    └── constants.ts
```

### React Component Pattern
```typescript
interface AgentCardProps {
  agentId: string;
  onSelect?: (id: string) => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({ agentId, onSelect }) => {
  const { agent, isLoading } = useAgent(agentId);
  
  if (isLoading) return <div>Loading...</div>;
  if (!agent) return null;
  
  return (
    <div onClick={() => onSelect?.(agentId)}>
      <h3>{agent.name}</h3>
    </div>
  );
};
```

### Zustand Store Pattern
```typescript
interface AgentStore {
  agents: Map<string, Agent>;
  addAgent: (agent: Agent) => void;
}

export const useAgentStore = create<AgentStore>((set) => ({
  agents: new Map(),
  addAgent: (agent) =>
    set((state) => ({
      agents: new Map(state.agents).set(agent.id, agent),
    })),
}));
```

### Python/FastAPI Patterns

**Type hints everywhere**:
```python
from typing import Optional
from pydantic import BaseModel

class AgentCreate(BaseModel):
    name: str
    personality: str
    max_tokens: int = 2048

async def get_agent(agent_id: str) -> Optional[Agent]:
    return await db.find_one({"id": agent_id})
```

**Error handling**:
```python
from fastapi import HTTPException

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = await agent_service.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
```

---

## Best Practices

1. **No type suppression**: Never use `as any`, `@ts-ignore`, or `type: ignore`
2. **No commits without request**: Only commit when explicitly asked
3. **Fix minimally**: When fixing bugs, don't refactor—fix the bug only
4. **Run diagnostics**: Use LSP diagnostics on changed files before marking complete
5. **Test new features**: Write tests for new functionality
6. **Match patterns**: Follow established conventions in the codebase
7. **Document public APIs**: Use JSDoc/docstrings for exported functions

## Git Commit Convention

```
<type>(<scope>): <subject>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(agent): add memory reflection system
fix(dialogue): handle null agent references
docs(readme): update setup instructions
```
