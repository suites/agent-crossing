# Agent Crossing: Neural Horizons - Technical Specification 🌳🤖

이 문서는 **Agent Crossing** 프로젝트의 기획 및 기술 명세서입니다. 개발자가 이 문서를 보고 시스템의 핵심 로직을 직접 구현할 수 있도록 상세히 정의합니다.

---

## 1. 프로젝트 개요
**Agent Crossing**은 각 NPC가 자신만의 의지와 기억을 가진 '생성형 에이전트(Generative Agents)'로 동작하는 2D 자율 사회 시뮬레이션입니다. 
본 프로젝트는 **Park et al. (2023)의 "Generative Agents: Interactive Simulacra of Human Behavior"** 논문의 아키텍처를 100% 반영하는 것을 목표로 합니다.

### 핵심 가치
- **자율성**: 고정된 스크립트 없이 NPC가 스스로 일과를 계획하고 행동함.
- **연속성**: 모든 경험은 메모리에 저장되어 미래의 행동과 대화에 영향을 미침.
- **창발성**: 에이전트 간의 상호작용을 통해 예상치 못한 사회적 현상(소문, 파티 등)이 발생함.

---

## 2. 에이전트 아키텍처 (The Brain)

에이전트의 인지 루프는 **인지(Perceive) -> 저장(Store) -> 계획(Plan) -> 행동(Act)**의 과정을 거칩니다.

### 2.1. 메모리 스트림 (Memory Stream)
에이전트의 모든 경험(관찰, 대화, 성찰)을 기록하는 시계열 데이터베이스입니다.
- **데이터 구조**: `(Content, Created_At, Last_Accessed, Importance, Type)`
- **중요도(Importance)**: 기억이 생성될 때 LLM이 1~10점 사이로 평가합니다. (예: "양치질을 했다" = 1, "수진이와 사귀기로 했다" = 10)

### 2.2. 회상 스코어러 (Retrieval Scorer)
현재 상황에서 가장 관련 있는 기억을 추출하기 위해 다음 세 가지 점수를 곱합니다.
**`Score = Recency · Importance · Relevance`**
- **Recency (최신성)**: `0.99 ^ hours_since_last_access`. 마지막 접근 이후 시간에 따라 지수적으로 감쇠합니다.
- **Importance (중요성)**: 저장된 중요도 점수 (1~10).
- **Relevance (관련성)**: 현재 상황(Query)과 기억 간의 벡터 코사인 유사도.

### 2.3. 성찰 시스템 (Reflection System)
단편적인 기억을 고차원적인 통찰(Insight)로 합성합니다.
- **트리거**: 새로 추가된 기억들의 `Importance` 합계가 **150점**을 초과할 때 실행.
- **프로세스**:
  1. 최근 기억(~100개)을 바탕으로 "이 에이전트의 상태에 대해 답할 수 있는 가장 핵심적인 질문 3개"를 생성.
  2. 각 질문에 대해 관련 기억을 회상(Retrieve).
  3. 회상된 기억들을 바탕으로 고차원 통찰(예: "지호는 수진이를 신뢰하고 있다")을 생성하여 메모리에 재저장.

### 2.4. 계층적 계획 (Planning Hierarchy)
에이전트의 행동 일관성을 유지하기 위해 계획을 단계별로 구체화합니다.
1. **Day Plan**: 기상 시 하루 전체의 거시적 일정 생성 (예: "09:00 도서관 출근, 18:00 카페 방문").
2. **Sector Plan**: 1시간 단위로 현재 위치와 구체적 목표 결정.
3. **Minute Plan**: 5~15분 단위의 구체적인 행동(Action) 수행.
4. **Re-planning (React)**: 중요도가 높은 새로운 관찰 발생 시, 현재 계획을 중단하고 즉시 계획을 수정할지 결정.

---

## 3. 환경 및 사회적 역학

### 3.1. 공간적 메모리 (Spatial Memory)
에이전트는 세상을 트리 구조로 인식합니다.
- **구조**: `Town -> Building -> Room -> Object` (예: "마을 -> 지호네 집 -> 주방 -> 냉장고")
- **이동**: 에이전트는 특정 물건을 찾을 때 이 트리 경로를 따라 위치를 파악하고 이동합니다.

### 3.2. 정보 전파 (Information Diffusion)
- **대화**: 에이전트 간 대화 시, Scorer에 의해 선택된 '중요하거나 상대방과 관련 있는 정보'를 공유합니다.
- **소문**: 한 에이전트가 목격한 사건이 대화를 통해 다른 에이전트에게 전달되며 마을 전체로 확산됩니다.

---

## 4. 기술 스택 및 시스템 구성

### 시스템 아키텍처
```
┌─────────────────────────────────────────┐
│  Web Browser (React 19 + Phaser 3)      │
│  - Canvas: 2D 타일맵 & NPC 렌더링         │
│  - UI: 대화창, 인스펙터, 관계 그래프       │
└─────────────────────────────────────────┘
           ↕ WebSocket (Real-time)
┌─────────────────────────────────────────┐
│  FastAPI Backend (Python)               │
│  - Agent Brain: LLM 추론 엔진             │
│  - Memory: Pinecone Vector DB           │
│  - World: 시간 및 상태 관리자              │
└─────────────────────────────────────────┘
```

### 상세 스택
- **Frontend**: Phaser 3 (Game), React 19 (UI), Tailwind CSS 4, Zustand (State)
- **Backend**: FastAPI, Transformers, MLX (Apple Silicon 가속), Pinecone (Vector DB)
- **Language Models**: Qwen 2.5-3B 또는 Phi-4-mini (Local SLM)

---

## 5. MVP 범위 (v0.1)
- 2명의 서로 다른 페르소나를 가진 에이전트.
- 기본적인 메모리 저장 및 회상 로직 (Score 공식 적용).
- 1개의 맵(마을 광장)에서의 자율 이동 및 대화.
- 플레이어의 관찰 및 대화 개입 기능.
