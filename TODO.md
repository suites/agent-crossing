# Agent Crossing Project TODO 📝

프로젝트 진행 현황과 앞으로의 작업 계획을 관리하는 문서입니다.

---

### 1단계: 환경 구축 및 POC 검증

- [x] 프로젝트 비전 및 기술 스택 확정 (`SPEC.md`, `AGENTS.md`)
- [x] **LLM 성능 검증**: Apple Silicon(M1 Pro) 환경에서 MLX 가속을 통한 Qwen 2.5-3B 추론 속도 확인 (평균 2.5초)
- [x] **Vector DB 검증**: Pinecone을 이용한 메모리 저장 및 검색 레이턴시 테스트 완료
- [x] **Monorepo 설정**: `pnpm workspace` 기반 `packages/shared`, `packages/frontend`, `packages/backend` 구조 구축
- [x] **Frontend 초기화**: Vite + React 19 + Phaser 3 기본 환경 설정 및 Canvas 통합
- [x] **Backend 초기화**: FastAPI 기본 서버 및 `AgentBrain` 기본 클래스(POC용) 구현

### 2단계: Generative Agents 핵심 엔진

- [ ] **AgentMemory 고도화**:
  - [ ] `MemoryEntry` 타입에 `importance`, `last_accessed` 필드 추가
  - [ ] `Recency * Importance * Relevance` 기반의 Retrieval Scorer 구현 (지수 감쇠 적용)
- [ ] **Cognitive 루프 구현**:
  - [ ] **Importance 평가**: `perceive` 시 LLM 호출로 중요도(1~10) 자동 채점
  - [ ] **Reflection 시스템**: 누적 점수 기반 트리거 및 Salient Questions 추출/Insight 생성
  - [ ] **계층적 Planning**: Day -> Sector -> Minute 계획 생성 로직 및 프롬프트 템플릿화
- [ ] **World Infrastructure**:
  - [ ] **World Clock**: 백엔드 중심의 시뮬레이션 시간(Tick) 시스템 구축
  - [ ] **Spatial Hierarchy**: 마을-건물-방-사물 구조의 트리 데이터 정의 및 에이전트 연동

### 3단계: 월드 통합 및 실시간 시각화

- [ ] **WebSocket API**: 에이전트의 상태(위치, 행동, 대화)를 실시간으로 브로드캐스팅
- [ ] **Phaser 엔진 고도화**:
  - [ ] **Map Loader**: Tiled 맵 또는 그리드 기반 환경 구성
  - [ ] **Pathfinding**: A\* 알고리즘을 이용한 NPC 이동 구현
  - [ ] **Action Executor**: 백엔드 계획을 프론트엔드 애니메이션/이동으로 변환
- [ ] **대화 UI**: React 19을 이용한 동적 말풍선 및 에이전트 인스펙터(현재 생각/기억 보기)

### 4단계: 사회적 상호작용 및 최적화

- [ ] **Information Diffusion**: 대화 중 중요 정보 선택 및 공유(소문 전파) 로직
- [ ] **Relationship Model**: 에이전트 간 호감도 및 관계 유형에 따른 대화 차별화
- [ ] **Performance**: 다수 에이전트(5인 이상) 동시 추론 시 병목 해결 및 배치 처리

---

## 🎯 주요 마일스톤

- **M1**: LLM/DB 기술 검증 완료 및 인프라 구축
- **M2**: 논문 기반 핵심 브레인(Memory, Plan, Reflect) 로직 완성
- **M3**: 브라우저 상에서 2명의 NPC가 자율적으로 생활하는 모습 시연
