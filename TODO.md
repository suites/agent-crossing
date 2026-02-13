# Agent Crossing Project TODO

이 문서는 Park et al. (2023) "Generative Agents: Interactive Simulacra of Human Behavior"(arXiv:2304.03442) 기준으로 정렬한 실행 TODO입니다.

---

## 0) Foundation (완료)

- [x] 프로젝트 구조 확정 (`packages/shared`, `packages/frontend`, `packages/backend`)
- [x] Frontend 기본 세팅 (React 19 + Phaser 3 + Zustand)
- [x] Backend 기본 세팅 (FastAPI + uv)
- [x] 로컬 LLM/벡터DB PoC 검증 (MLX/Vector DB)
- [ ] 메모리 저장소 전환: Pinecone -> PostgreSQL + pgvector

## 1) Memory Stream & Retrieval (논문 핵심 1)

- [ ] MemoryObject 스키마 정의
  - [x] 기본 메모리 객체/스트림 구현 (`packages/backend/src/memory/memory_object.py`, `packages/backend/src/memory/memory_stream.py`)
  - [x] 필수 필드 1차 구현: `id`, `description`, `creation_timestamp`, `last_accessed`, `importance`, `node_type`
  - [x] 확장 필드 일부 구현: `citations`, `embedding`
  - [ ] 스키마 명세 동기화: `description/creation_timestamp` vs `content/created_at` 네이밍 통일
  - [ ] `location_path` 필드 추가 (예: `Town > House > Kitchen > Stove`)
  - [ ] `node_type` 확장 여부 결정: 현재 `OBSERVATION | REFLECTION | PLAN` + 필요 시 `CONVERSATION`
- [ ] Importance scoring 구현
  - [ ] 기억 생성 시 LLM으로 `1~10` 정수 중요도 산정
  - [ ] 파싱 실패 fallback 규칙 정의 (예: 기본값 3)
- [ ] Retrieval scoring 구현 (가중 합)
  - [ ] 공식: `score = alpha * recency + beta * importance + gamma * relevance`
  - [ ] 기본 가중치: `alpha = beta = gamma = 1.0`
  - [ ] 각 항목 Min-Max 정규화 `[0,1]`
  - [ ] Recency decay: `0.995 ^ hours_since_last_access`
  - [ ] Relevance: query embedding과 memory embedding cosine similarity
- [ ] Retriever 품질 검증
  - [x] `add_memory` 단위 테스트 작성 (`packages/backend/tests/test_memory_stream.py`)
  - [ ] 동일 query 재호출 시 top-k 일관성 체크
  - [ ] 중요 이벤트(importance high)가 retrieval에서 상위 노출되는지 테스트

## 2) Reflection Loop (논문 핵심 2)

- [ ] Reflection trigger 구현
  - [ ] 최근 이벤트 누적 중요도 `>= 150` 시 reflection 실행
  - [ ] reflection 실행 후 누적값 리셋 정책 정의
- [ ] Reflection 생성 파이프라인
  - [ ] 최근 기억 100개 기반 salient questions 3개 생성
  - [ ] 질문별 관련 기억 retrieval
  - [ ] high-level insight 5개 생성
  - [ ] insight에 근거 memory id를 `citations`로 연결
- [ ] Reflection-on-reflection 허용
  - [ ] reflection memory도 retrieval 후보에 포함

## 3) Planning & Re-planning (논문 핵심 3)

- [ ] 계층형 계획 생성
  - [ ] Day plan (5~8개 broad strokes)
  - [ ] Hourly plan
  - [ ] Minute plan (5~15분)
  - [ ] 각 plan에 `start_time`, `duration`, `location` 포함
- [ ] Tick 단위 react 판단
  - [ ] 관찰 이벤트가 현재 계획을 방해/우선하는지 판정
  - [ ] 반응 필요 시 현재 시점 이후 계획만 재수립
- [ ] 대화 연계 planning
  - [ ] 타 에이전트 조우 시 pass-by vs converse decision
  - [ ] 대화 후 새 정보/관계 업데이트를 plan에 반영

## 4) World Integration (시뮬레이션 완성)

- [ ] Backend real-time state pipeline
  - [ ] WebSocket broadcast: position/action/dialogue/emoji/current plan
  - [ ] world clock + tick scheduler
- [ ] Frontend visualization
  - [ ] Tiled map + collision
  - [ ] A\* pathfinding
  - [ ] agent inspector (memory/plan/reflection 뷰)
  - [ ] God mode 입력 -> perception event 주입

## 5) Social Dynamics & Evaluation (논문 검증)

- [ ] 정보 확산 실험 구현
  - [ ] seed fact 주입 후 인지한 agent 비율 측정
- [ ] 관계 형성 지표 구현
  - [ ] 네트워크 밀도 `eta = 2|E| / (|V|(|V|-1))` 계산
- [ ] 협업/조율 지표 구현
  - [ ] 이벤트 초대 대비 실제 도착 agent 수 측정
- [ ] Interview evaluator 구현 (총 25문항)
  - [ ] 카테고리: self-knowledge, memory, plans, reactions, reflections
- [ ] Ablation 실험 플래그
  - [ ] no-observation / no-reflection / no-planning 비교 모드

---

## Milestones

- [x] M1: Infra & PoC 완료
- [ ] M2: Single-agent believable daily life
- [ ] M3: Two-agent social interaction + information diffusion
- [ ] M4: Multi-agent town simulation + user intervention

## Notes

- [ ] `SPEC.md`의 retrieval 식을 논문 기준(`weighted sum`)으로 동기화
- [ ] MVP 기준은 "보이는 연출"보다 "인지 루프 정확성"을 우선
