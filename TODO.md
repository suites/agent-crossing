# Agent Crossing Project TODO

이 문서는 Park et al. (2023) "Generative Agents: Interactive Simulacra of Human Behavior"(arXiv:2304.03442) 기반으로 정렬한 실행 TODO입니다.

## 사용 방법

- 우선순위: `P0`(지금 당장), `P1`(다음), `P2`(후순위)
- 체크 기준: 각 항목의 `DoD`(Definition of Done)를 만족하면 `[x]`
- 의존성: `Depends on`이 완료되기 전에는 다음 항목 시작 금지
- 기준 문서: 알고리즘/상수는 `SPEC.md` + `AGENTS.md`를 기준으로 동기화

---

## 0) Foundation

- [x] 프로젝트 구조 확정 (`packages/shared`, `packages/frontend`, `packages/backend`)
- [x] Frontend 기본 세팅 (React 19 + Phaser 3 + Zustand)
- [x] Backend 기본 세팅 (FastAPI + uv)
- [x] 로컬 LLM/벡터DB PoC 검증 (MLX/Vector DB)
- [x] `P0` 메모리 저장소 전환 완료 상태 확정: PostgreSQL + pgvector

## 1) Memory Stream & Retrieval (논문 핵심 1)

- [x] `P0` MemoryObject 데이터 모델 명세 동기화
  - Depends on: 없음
  - DoD:
    - [x] 기본 메모리 객체/스트림 구현 (`packages/backend/src/memory/memory_object.py`, `packages/backend/src/memory/memory_stream.py`)
    - [x] 필수 필드 1차 구현: `id`, `content`, `created_at`, `last_accessed`, `importance`, `node_type`
    - [x] 확장 필드 일부 구현: `citations`, `embedding`
    - [x] 필드 네이밍 통일 (`content/creation_timestamp` vs `content/created_at`)

- [ ] `P0` Importance scoring 구현
  - Depends on: MemoryObject 데이터 모델 명세 동기화
  - DoD:
    - [ ] 기억 생성 시 LLM으로 `1~10` 정수 중요도 산정
    - [ ] 파싱 실패 fallback 규칙 구현 (예: 기본값 3)
    - [ ] 단위 테스트 추가

- [ ] `P0` Retrieval scoring 구현 (가중합)
  - Depends on: MemoryObject 데이터 모델 명세 동기화
  - DoD:
    - [ ] 공식: `score = alpha * recency + beta * importance + gamma * relevance`
    - [ ] 기본 가중치: `alpha = beta = gamma = 1.0`
    - [ ] 각 항목 Min-Max 정규화 `[0,1]`
    - [ ] Recency decay: `0.995 ** hours_since_last_access`
    - [ ] Relevance: query embedding과 memory embedding cosine similarity
    - [ ] 단위 테스트 추가

- [ ] `P0` Retriever 품질 검증
  - Depends on: Retrieval scoring 구현
  - DoD:
    - [x] `add_memory` 단위 테스트 작성 (`packages/backend/tests/test_memory_stream.py`)
    - [ ] 동일 query 재호출 시 top-k 일관성 체크
    - [ ] 중요 이벤트(importance high)가 retrieval 상위 노출되는지 검증
    - [ ] `pnpm test:backend` 또는 `uv run pytest` 통과

## 2) Reflection Loop (논문 핵심 2)

- [ ] `P1` Reflection trigger 구현
  - Depends on: Retrieval scoring 구현
  - DoD:
    - [ ] 최근 이벤트 누적 중요도 `>= 150` 시 reflection 실행
    - [ ] reflection 실행 후 누적값 리셋 정책 구현
    - [ ] 임계치 테스트 추가

- [ ] `P1` Reflection 생성 파이프라인
  - Depends on: Reflection trigger 구현
  - DoD:
    - [ ] 최근 기억 100개 기반 salient questions 3개 생성
    - [ ] 질문별 관련 기억 retrieval
    - [ ] high-level insight 5개 생성
    - [ ] insight의 근거 memory id를 `citations`로 연결

- [ ] `P1` Reflection-on-reflection 허용
  - Depends on: Reflection 생성 파이프라인
  - DoD:
    - [ ] reflection memory를 retrieval 후보에 포함
    - [ ] 회귀 테스트 추가

## 3) Planning & Re-planning (논문 핵심 3)

- [ ] `P1` 계층형 계획 생성
  - Depends on: Reflection Loop 핵심 완료
  - DoD:
    - [ ] Day plan (5~8 broad strokes)
    - [ ] Hourly plan
    - [ ] Minute plan (5~15분)
    - [ ] 각 plan에 `start_time`, `duration`, `location` 포함

- [ ] `P1` Tick 단위 react 판단
  - Depends on: 계층형 계획 생성
  - DoD:
    - [ ] 관찰 이벤트가 현재 계획을 방해/우선하는지 판정
    - [ ] 반응 시 현재 시점 이후 계획만 재수립

- [ ] `P2` 대화 연계 planning
  - Depends on: Tick 단위 react 판단
  - DoD:
    - [ ] 타 에이전트 조우 시 pass-by vs converse decision
    - [ ] 대화 후 새 정보/관계 업데이트를 plan에 반영

## 4) World Integration (시뮬레이션)

- [ ] `P1` Backend real-time state pipeline
  - Depends on: Planning & Re-planning 핵심 완료
  - DoD:
    - [ ] WebSocket broadcast: position/action/dialogue/emoji/current plan
    - [ ] world clock + tick scheduler

- [ ] `P2` Frontend visualization
  - Depends on: Backend real-time state pipeline
  - DoD:
    - [ ] Tiled map + collision
    - [ ] A\* pathfinding
    - [ ] agent inspector (memory/plan/reflection 뷰)
    - [ ] God mode 입력 -> perception event 주입

## 5) Social Dynamics & Evaluation (논문 검증)

- [ ] `P2` 정보 확산 실험
  - Depends on: 대화 연계 planning
  - DoD:
    - [ ] seed fact 주입 후 인지한 agent 비율 측정

- [ ] `P2` 관계 형성 지표
  - Depends on: 정보 확산 실험
  - DoD:
    - [ ] 네트워크 밀도 `eta = 2|E| / (|V|(|V|-1))` 계산

- [ ] `P2` 협업/조율 지표
  - Depends on: 관계 형성 지표
  - DoD:
    - [ ] 이벤트 초대 대비 실제 도착 agent 수 측정

- [ ] `P2` Interview evaluator (총 25문항)
  - Depends on: Reflection Loop 핵심 완료
  - DoD:
    - [ ] 카테고리: self-knowledge, memory, plans, reactions, reflections
    - [ ] 자동 채점/결과 저장 포맷 정의

- [ ] `P2` Ablation 실험 플래그
  - Depends on: Interview evaluator
  - DoD:
    - [ ] no-observation / no-reflection / no-planning 비교 모드

---

## Milestones

- [x] M1: Infra & PoC 완료
- [ ] M2: Single-agent believable daily life
  - 조건: 1) Memory/Retrieval P0 완료 + 2) Reflection P1 완료 + 3) Planning P1 완료
- [ ] M3: Two-agent social interaction + information diffusion
  - 조건: 대화 연계 planning + 정보 확산 실험
- [ ] M4: Multi-agent town simulation + user intervention
  - 조건: World integration + Social/Evaluation 핵심 항목 완료
