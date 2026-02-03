# Agent Crossing Development Guide

**AI 에이전트를 위한 개발 지침**

**Tech Stack**: TypeScript, React 19, Tailwind CSS 4, Zustand, Phaser 3, FastAPI, Transformers, ChromaDB, pnpm  
**Status**: Greenfield project (specification phase)

---

## 🎯 Core Principles

1. **Greenfield project** - 아직 코드가 없음. 패키지 설정, 빌드 시스템 등은 Week 2에 구성됨
2. **Brain-first approach** - AI 에이전트 브레인 검증 → 프론트엔드 구현 순서
3. **pnpm monorepo** - packages/shared, packages/frontend, packages/backend 구조 예정

---

## 📦 Technology Stack

### Frontend
- **Phaser 3**: 웹 기반 2D 게임 엔진 (Canvas/WebGL)
- **React 19**: UI 컴포넌트 오버레이
- **Tailwind CSS 4**: 스타일링
- **Zustand**: 상태 관리 (Phaser ↔ React 브릿지)
- **TypeScript**: 정적 타입 체크
- **pnpm**: 패키지 매니저 (monorepo)

### Backend
- **FastAPI**: 비동기 Python 웹 프레임워크
- **Transformers**: Hugging Face 모델 로더
- **Qwen 2.5-3B or Phi-4-mini**: 로컬 SLM
- **ChromaDB**: Vector database (메모리 저장소)
- **Python 3.11+**: 런타임
- **WebSocket**: 실시간 통신 (Frontend ↔ Backend)

---

## 📋 Code Style Guidelines

### Import Organization

**Order** (blank lines between groups):
```
1. External dependencies (node_modules)
2. Internal absolute imports (@/ alias)
3. Relative imports (./ or ../)
4. Type imports
```

**Rules**:
- Use `@/` for cross-module imports
- Use `./` only within same module
- Alphabetize within each group
- One import per line

### Naming Conventions

**Files**:
```
AgentCard.tsx              # Components (PascalCase)
agent.service.ts           # Services (camelCase + .service)
agent.store.ts             # Stores (camelCase + .store)
useAgent.ts                # Hooks (camelCase with 'use' prefix)
agent.types.ts             # Types (camelCase + .types)
vector.util.ts             # Utils (camelCase + .util)
agent.service.test.ts      # Tests (source name + .test)
```

**Code**:
```
calculateAgentAffinity()   # Functions: camelCase, verb-noun
agentList                  # Variables: camelCase
MAX_MEMORY_TOKENS          # Constants: UPPER_SNAKE_CASE
AgentBrain                 # Classes: PascalCase
Agent                      # Interfaces/Types: PascalCase
```

### TypeScript Rules

**Critical**:
- ❌ **NEVER use `any`, `as any`, `@ts-ignore`, `@ts-expect-error`**
- ✅ Always use explicit types for function parameters and return values
- ✅ Use strict null checks (`Type | null`, `Type | undefined`)
- ✅ Use optional chaining (`?.`) and nullish coalescing (`??`)

**React 19**:
- ❌ Avoid `React.FC` (deprecated pattern)
- ✅ Use plain function signatures: `function ComponentName(props: Props) {}`

**Python**:
- ✅ Type hints everywhere (`from typing import Optional, List`)
- ✅ Use Pydantic models for validation
- ✅ Async/await with FastAPI

### Error Handling

**TypeScript**:
- Use try/catch for async operations
- Consider Result type pattern for critical paths (only when needed)
- Never suppress errors silently

**Python**:
- Use FastAPI's HTTPException for API errors
- Type hint exceptions in function signatures
- Log errors with context

### File Organization

```
packages/
├── shared/              # Shared types, constants, utils
│   └── src/
│       ├── types/       # Cross-package type definitions
│       └── constants/
├── frontend/            # React + Phaser
│   └── src/
│       ├── game/        # Phaser scenes, sprites
│       ├── components/  # React UI components (UI only)
│       ├── stores/      # Zustand state management
│       ├── hooks/       # Custom React hooks
│       ├── services/    # Business logic & API calls
│       ├── types/       # Frontend-specific types
│       └── utils/       # Pure utility functions
└── backend/             # FastAPI
    └── src/
        ├── agents/      # Agent cognitive engine
        ├── memory/      # Vector store (ChromaDB)
        ├── api/         # FastAPI endpoints
        └── models/      # Pydantic models
```

**Module Boundaries**:
- **components/**: UI only, no business logic
- **services/**: Business logic, API calls
- **stores/**: Global state
- **utils/**: Pure functions, no side effects

---

## 🤖 AI Agent Guidelines

### Before Starting Work

1. **Read GETTING_STARTED.md first** - 전체 프로젝트 계획 및 우선순위 파악
2. **Check project status** - Greenfield이므로 기존 패턴 참고 불가
3. **Verify tech stack** - 위 Technology Stack 섹션 확인

### During Development

1. **No type suppression**: Never use `any`, `as any`, `@ts-ignore`, `type: ignore`
2. **No commits without explicit request**: Only commit when user asks
3. **Fix minimally**: When fixing bugs, don't refactor—fix the bug only
4. **Run diagnostics**: Use `mcp_lsp_diagnostics` on changed files before marking complete
5. **Match planned structure**: Follow file organization above when creating new files
6. **Test new features**: Write tests for new functionality
7. **Document public APIs**: Use JSDoc (TypeScript) or docstrings (Python)

### Phaser 3 Guidelines

**Scene Organization**:
- One scene per file
- Use TypeScript classes extending `Phaser.Scene`
- Keep game logic separate from rendering

**Sprite Management**:
- Use Phaser's built-in sprite pooling for performance
- Store agent state in Zustand, not Phaser objects

**Integration with React**:
- Phaser runs in `<canvas>` element
- React handles UI overlays (dialogue, menus)
- Zustand bridges state between Phaser and React
- Never manipulate DOM from Phaser

### ChromaDB / Vector Memory

- Use semantic search for agent memory retrieval
- Limit memory corpus per agent (<1000 entries for MVP)
- Test retrieval latency early (target: <100ms for top-5)

### LLM Integration

- All LLM calls happen in Python backend
- Frontend never directly calls LLM
- Use streaming for long responses (WebSocket)
- Add timeout handling (default: 30s)

---

## 📝 Git Commit Convention

```
<type>(<scope>): <subject>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(agent): add memory retrieval system
fix(dialogue): handle null agent references
docs(readme): update setup instructions
chore(deps): update Phaser to 3.80.0
```

**Rules**:
- Use lowercase for type and scope
- Subject in imperative mood ("add" not "added")
- No period at the end
- Keep subject under 72 characters

---

## ⚠️ Important Notes

### What Doesn't Exist Yet
- ❌ No package.json / pyproject.toml
- ❌ No build scripts or commands
- ❌ No existing codebase to reference
- ❌ No established patterns

### What to Do First (Week 1-2)
1. Backend POC: LLM + ChromaDB validation
2. Monorepo setup: pnpm workspace
3. Frontend bootstrap: Vite + React + Phaser basic scene

Refer to **GETTING_STARTED.md** for detailed implementation roadmap.

---

## 🔄 When to Update This Document

Update this file when:
- Tech stack changes (new dependencies)
- New architectural patterns emerge from real code
- Team decides on new conventions
- Phaser-specific patterns are established

Do NOT update for:
- Individual code changes
- Bug fixes
- Feature additions

Keep this document focused on **guidelines**, not **implementation details**.
