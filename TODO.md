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

### 2단계: Generative Agents 핵심 엔진 (현재 단계)

- [ ] **AgentMemory 고도화**: 단순 검색에서 논문의 `Recency * Importance * Relevance` 스코어링 로직으로 전환
- [ ] **Importance 평가 로직**: 기억 생성 시 LLM을 호출하여 중요도(1~10)를 자동 채점하는 기능 추가
- [ ] **Reflection 루프 구현**: 누적 중요도 150점 달성 시 Salient Questions를 통한 성찰 기능 구현
- [ ] **계층적 Planning**: Day Plan -> Sector Plan -> Minute Plan 생성 로직 구현

### 3단계: 월드 및 사회적 상호작용

- [ ] **Spatial Memory**: 장소/사물을 트리 구조로 정의하고 에이전트가 이를 인식하게 함
- [ ] **WebSocket 통합**: 백엔드의 에이전트 행동 결정을 프론트엔드로 실시간 전송
- [ ] **대화 UI**: React 오버레이를 이용한 NPC 말풍선 및 대화 시스템

### 4단계: 시뮬레이션 고도화 (v0.2+)

- [ ] **Information Diffusion**: 소문 전파 알고리즘 구현
- [ ] **Affinity Matrix**: 에이전트 간 관계 점수 시각화 및 동적 변화
- [ ] **다수 에이전트 최적화**: 5인 이상의 에이전트가 동시에 동작할 수 있는 스케줄링 최적화

---

## 🎯 주요 마일스톤

- **M1**: LLM/DB 기술 검증 완료 및 인프라 구축
- **M2**: 논문 기반 핵심 브레인(Memory, Plan, Reflect) 로직 완성
- **M3**: 브라우저 상에서 2명의 NPC가 자율적으로 생활하는 모습 시연
