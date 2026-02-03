# Agent Crossing 시작 가이드

**AI + 사람이 볼 프로젝트 계획 문서**

---

## 🌟 프로젝트 비전

**Agent Crossing: Neural Horizons**는 각 NPC가 자신만의 의지와 기억을 가진 생성형 에이전트로 동작하는 2D 자율 사회 시뮬레이션입니다.

- **Living Society**: 고정된 스크립트 없이 NPC들이 스스로 스케줄을 짜고 관계를 맺으며 살아가는 마을
- **Emergent Storytelling**: 호감도, 소문, 성찰이 결합되어 매번 새로운 서사 발생
- **Privacy-First AI**: 로컬 SLM 사용으로 사용자 데이터 외부 유출 없음

---

## 🎯 핵심 원칙

**"먼저 두뇌를 만들고, 나중에 몸을 만들어라"**

- 최대 리스크: 로컬 SLM 추론 속도 (Qwen 2.5-3B, Phi-4-mini)
- UI보다 AI 에이전트 브레인 검증 우선
- 에이전트의 지능적 행동이 핵심 가치

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────┐
│  웹 브라우저 (React App)                  │
│  ┌─────────────────────────────────┐    │
│  │  Phaser 3 Canvas                │    │
│  │  - 2D 타일맵                     │    │
│  │  - NPC 스프라이트                │    │
│  │  - 경로 탐색, 애니메이션          │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  React UI 오버레이               │    │
│  │  - 대화 말풍선                   │    │
│  │  - 에이전트 인스펙터              │    │
│  │  - 관계 그래프                   │    │
│  └─────────────────────────────────┘    │
│           ↕ Zustand Store               │
└─────────────────────────────────────────┘
          ↕ WebSocket
┌─────────────────────────────────────────┐
│  FastAPI 백엔드 (Python)                 │
│  - Agent Brain (추론 엔진)               │
│  - Memory Store (Vector DB)             │
│  - World State Manager                  │
└─────────────────────────────────────────┘
```

### 에이전트 시스템

**인지 엔진 (Agent Brain)**:
- Memory & Retrieval: 대화/관찰을 Vector DB에 저장, 유사도 기반 회상
- Reflection System: 고차원 인식 생성 ("A는 친절한 사람이다")
- Planning & Execution: 24시간 일과 계획, 이벤트 발생 시 Re-planning

**사회적 역학**:
- Affinity Matrix: 호감도/신뢰도/친밀도로 관계 관리
- Gossip Logic: 목격 사건을 친밀한 NPC에게 전파

**게임플레이**:
- Autonomous Interaction: NPC 간 자율 대화 및 이동
- Dynamic Dialogue: 페르소나 유지하며 자유 질의응답
- Influence & Observation: 플레이어 참여 또는 관찰 모드

---

## 📋 실행 단계

| # | 작업 | 예상 시간 | 목적 |
|---|------|----------|------|
| 1 | LLM 추론 속도 검증 | 2-4h | 3초 초과 시 아키텍처 재검토 |
| 2 | CLI 에이전트 브레인 POC | 1-2d | UI 없이 핵심 로직 검증 |
| 3 | Vector DB 벤치마크 | 4h | ChromaDB vs Qdrant vs LanceDB |
| 4 | Phaser 3 프로토타입 | 4-6h | 기본 상호작용 테스트 |
| 5 | Monorepo 설정 | 1-2h | pnpm workspace 구성 |

**타임라인**: Week 1 (1-3) → Week 2 (4-5)

---

## 📅 Week 1 체크리스트

### Day 1-2: LLM 검증
- [ ] Qwen 2.5-3B 로드 및 100 토큰 생성 (목표: <3초)
- [ ] 한국어 품질 테스트
- [ ] 레이턴시 수용 가능 여부 결정

### Day 3: Vector DB
- [ ] ChromaDB 설치 및 초기화
- [ ] AgentMemory 클래스 (add/retrieve)
- [ ] 100개 메모리 top-5 검색 벤치마크

### Day 4-5: CLI 에이전트 브레인
- [ ] AgentBrain 클래스 (perceive/decide/converse)
- [ ] 2개 페르소나로 10턴 대화 테스트
- [ ] 일관성, 메모리 회상, 성격 유지 평가

### Day 6-7: Frontend + Monorepo
- [ ] Phaser 3 기본 씬 (2개 스프라이트)
- [ ] Phaser + React 통합 테스트
- [ ] pnpm workspace 설정

---

## 🎮 MVP 범위

### ✅ 포함
- 2개 에이전트 (다른 성격)
- 기본 메모리 (최근 20개 상호작용)
- 간단한 일과 (기상 → 일 → 식사 → 사교 → 수면)
- 1개 장소 (마을 광장)
- 에이전트 자율 대화
- 플레이어 관찰 및 대화 참여

### ❌ v0.2 이후
- 성찰 시스템
- 소문 전파
- 복잡한 경로 탐색
- 호감도 매트릭스 동적 변화
- 모순 감지

---

## 🚨 리스크

| 리스크 | 심각도 | 대응 |
|--------|--------|------|
| LLM 레이턴시 | 🔴 | Day 1 검증. 3초+ 시: 배치 추론/llama.cpp/턴제 |
| 메모리 검색 속도 | 🔴 | Vector DB 벤치마크. MVP는 <1000개/에이전트 |
| 에이전트 조정 | 🟡 | 중앙 World Tick 시스템 |
| 프롬프트 엔지니어링 | 🟡 | 2-3일 튜닝 할애 |

---

## 🛠️ 기술 스택

**Frontend**: Phaser 3, React 19, Tailwind CSS 4, Zustand, TypeScript  
**Backend**: FastAPI, Transformers, Qwen 2.5-3B/Phi-4-mini, ChromaDB, Python 3.11+  
**DevOps**: pnpm (monorepo), WebSocket

---

## ✅ 성공 기준

**Week 1**: LLM <3초, CLI 2개 에이전트 10턴 대화, 메모리 회상 정확  
**MVP (Week 4-6)**: 브라우저에서 2개 에이전트 자율 행동, 대화, 메모리 저장/회상, 성격 유지

---

## 📝 메모

- GPU 없으면: llama.cpp 사용 (CPU 최적화)
- 2개 에이전트만: O(N²) 복잡도, 핵심부터 완성
- 레이턴시 대응: "생각 중..." 애니메이션
