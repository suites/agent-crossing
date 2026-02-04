# Agent Crossing Development Guide (for AI Agents) 🤖

이 문서는 **Sisyphus** 및 모든 하위 에이전트가 코드를 작성하고 시스템을 고도화할 때 반드시 준수해야 하는 지침입니다.

---

## 🎯 핵심 원칙 (Core Principles)
1. **Brain-First**: 시각적 효과보다 에이전트의 지능적 행동(논문 아키텍처 준수)을 우선함.
2. **Strict Typing**: TypeScript와 Python Type Hint를 엄격히 사용하여 런타임 오류를 최소화함.
3. **No Slop**: senior engineer 수준의 정밀하고 일관된 코드를 작성함.

---

## 📂 프로젝트 구조 및 경계 (Module Boundaries)

```
packages/
├── shared/              # 공통 타입 및 상수 (Frontend/Backend 공유)
├── frontend/            # React 19 + Phaser 3 (UI 및 시각화)
│   ├── src/game/        # Phaser Scene 및 Sprite 로직
│   ├── src/components/  # React UI 컴포넌트 (UI Only)
│   └── src/stores/      # Zustand 상태 관리 (Game ↔ UI 브릿지)
└── backend/             # FastAPI (인지 엔진 및 데이터 관리)
    ├── src/agents/      # AgentBrain 및 인지 루프 로직
    ├── src/memory/      # Vector DB 및 Retrieval Scorer
    └── src/api/         # FastAPI 엔드포인트 및 WebSocket
```

- **중요**: Phaser 로직에서 직접 DOM을 조작하지 말 것. 모든 상태는 Zustand를 통해 React와 통신함.

---

## 📋 코딩 규칙 (Coding Standards)

### TypeScript (React 19)
- **JSX**: `react-jsx` 방식 사용 (파일 상단 `import React` 불필요).
- **Types**: ❌ `any`, `as any`, `@ts-ignore` 절대 금지. ✅ 명시적 Interface 정의.
- **Components**: `React.FC` 대신 일반 함수 선언문 사용.

### Python (FastAPI)
- **Type Hints**: 모든 함수 매개변수와 반환값에 타입 힌트 필수.
- **Validation**: Pydantic 모델을 사용하여 API 데이터 검증.
- **Async**: FastAPI의 장점을 살리기 위해 비동기(async/await) 패턴 유지.

---

## 🤖 에이전트 구현 가이드 (Generative Agents Spec)

### LLM 호출 시 필수 컨텍스트 (Prompting)
에이전트가 결정을 내리거나 대화할 때, 프롬프트에는 반드시 다음 항목이 포함되어야 합니다:
1. **Persona**: 에이전트의 이름, 성격, 핵심 가치관.
2. **Current Plan**: 현재 수행 중인 세부 단계 (Minute Plan).
3. **Retrieved Memories**: Retrieval Scorer에 의해 선택된 관련 기억들.
4. **Spatial Context**: 현재 위치의 계층 정보 (예: "지호네 집 > 거실").

### 메모리 및 성찰 로직
- **Retrieval**: 단순 벡터 검색 금지. 반드시 `Recency * Importance * Relevance` 공식을 구현할 것.
- **Reflection**: `accumulated_importance >= 150`일 때 `reflect()` 메서드가 자발적으로 실행되도록 설계.

---

## 📝 작업 절차 및 컨벤션
- **Progress Tracking**: `TODO.md`를 바탕으로 진행 상황을 파악하고, 작업이 완료되면 반드시 체크(`[x]`) 처리함.
- **Specification Reference**: 모든 기능 구현 및 세부 로직은 `SPEC.md`를 최우선 참고하여 구현함.
- **Git Commit**: `<type>(<scope>): <subject>` 형식을 준수 (예: `feat(agent): add reflection loop`).
- **Verification**: 파일 수정 후 반드시 `lsp_diagnostics`를 실행하여 오류 확인.
- **Minimal Fix**: 버그 수정 시 리팩토링을 병행하지 말고 최소한의 코드만 수정함.
