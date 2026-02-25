# Agent Crossing Project TODO

논문(Generative Agents, 2023) 스펙을 위에서 아래로 읽으면서,
기능을 작은 단위로 하나씩 구현하기 위한 실행 보드.

## 사용 규칙

- 우선순위: `P0`(핵심), `P1`(핵심 확장), `P2`(검증/고도화)
- 순서 원칙: 같은 섹션에서는 위에서 아래 순서로만 진행
- 체크 기준: 항목 아래 `DoD`를 모두 만족하면 `[x]`
- 의존성: `Depends on`이 완료되기 전에는 시작하지 않음
- 기준 문서: 알고리즘/상수/계약은 `SPEC.md` + `AGENTS.md`와 동기화

---

## 0) Foundation

- [x] 프로젝트 기본 골격 확정 (`packages/shared`, `packages/frontend`, `packages/backend`)
- [x] Frontend 런타임 부팅 (React 19 + Phaser 3 + Zustand)
- [x] Backend 런타임 부팅 (FastAPI + uv)
- [x] 로컬 LLM + 벡터 DB PoC 통과 (MLX/Vector DB)
- [x] `P0` 메모리 영속 스토어 전환 확정 (PostgreSQL + pgvector)

## 1) Memory Stream & Retrieval (논문 핵심 1)

- [x] `P0` MemoryObject 스키마를 코드와 1:1로 맞춘다
  - Depends on: 없음
  - DoD:
    - [x] 기본 메모리 객체/스트림을 구현한다 (`packages/backend/src/memory/memory_object.py`, `packages/backend/src/memory/memory_stream.py`)
    - [x] 필수 필드(`id`, `content`, `created_at`, `last_accessed`, `importance`, `node_type`)를 저장/조회 경로에 반영한다
    - [x] 확장 필드(`citations`, `embedding`)를 지원한다
    - [x] 필드 이름 충돌(`content/creation_timestamp` vs `content/created_at`)을 하나로 정리한다

- [x] `P0` Importance scoring을 안정적으로 계산한다
  - Depends on: MemoryObject 스키마 동기화
  - DoD:
    - [x] 기억 생성 시 LLM 출력으로 중요도를 `1~10` 정수로 산정한다
    - [x] 파싱 실패 시 fallback 규칙(기본값 3)을 적용한다
    - [x] 단위 테스트로 점수 범위/실패 경로를 검증한다

- [x] `P0` Retrieval score(가중합)를 공식대로 계산한다
  - Depends on: MemoryObject 스키마 동기화
  - DoD:
    - [x] Recency decay를 `0.995 ** hours_since_last_access`로 계산한다
    - [x] Relevance를 query embedding vs memory embedding cosine similarity로 계산한다
    - [x] 최종 점수를 `score = (alpha * recency) + (beta * importance) + (gamma * relevance)`로 계산한다
    - [x] 기본 가중치 `alpha = beta = gamma = 1.0`을 적용한다
    - [x] 각 항목을 Min-Max로 `[0,1]` 정규화한다
    - [x] 단위 테스트로 계산식/정규화 edge case를 검증한다

- [x] `P0` Retriever 품질을 테스트로 고정한다
  - Depends on: Retrieval scoring 구현
  - DoD:
    - [x] `add_memory` 단위 테스트를 작성한다 (`packages/backend/tests/test_memory_stream.py`)
    - [x] 동일 query 재호출 시 top-k 결과가 일관적인지 확인한다
    - [x] 중요 이벤트(high importance)가 retrieval 상위에 노출되는지 확인한다
    - [x] `pnpm test:backend` 또는 `uv run pytest`가 통과한다

## 2) Reflection Loop (논문 핵심 2)

### 2-A. Trigger와 누적값 관리

- [] `P1` Reflection trigger를 임계치 기반으로 동작시킨다
  - Depends on: Retrieval scoring 구현
  - DoD:
    - [x] 최근 이벤트 누적 중요도 `>= 150`에서 reflection을 실행한다
    - [x] reflection 실행 후 누적값 리셋 정책을 적용한다
    - [] 임계치 테스트를 추가한다

### 2-B. Reflection 생성 파이프라인 세분화

- [ ] `P1` reflection 입력 메모리 윈도우를 고정한다
  - Depends on: Reflection trigger 구현
  - DoD:
    - [x] 최근 기억 100개를 시간 역순으로 수집한다
    - [] 기억 부족 시 가능한 개수만 사용하고 실패 없이 진행한다

- [x] `P1` salient question 3개를 생성한다
  - Depends on: reflection 입력 메모리 윈도우 고정
  - DoD:
    - [x] LLM 출력에서 질문 3개를 안정적으로 파싱한다
    - [x] 파싱 실패 시 fallback 질문 생성 규칙을 적용한다

- [x] `P1` 질문별 관련 기억 retrieval을 수행한다
  - Depends on: salient question 3개 생성
  - DoD:
    - [x] 질문마다 관련 기억 top-k를 조회한다
    - [x] 질문별 retrieval 결과를 구분된 구조로 유지한다

- [x] `P1` high-level insight 5개를 생성한다
  - Depends on: 질문별 관련 기억 retrieval 수행
  - DoD:
    - [x] insight 5개를 생성하고 빈 항목 없이 저장한다
    - [] insight마다 source question을 연결한다

- [ ] `P1` insight citations를 memory id로 연결한다
  - Depends on: high-level insight 5개 생성
  - DoD:
    - [ ] insight별 근거 memory id 목록을 `citations`에 저장한다
    - [ ] 존재하지 않는 memory id가 citations에 들어가지 않도록 검증한다

### 2-C. Reflection 재귀 활용

- [ ] `P1` reflection memory를 retrieval 후보에 포함한다
  - Depends on: insight citations 연결
  - DoD:
    - [ ] node_type=REFLECTION 메모리가 retrieval 후보군에 포함된다
    - [ ] observation/reflection 혼합 시 점수 계산이 깨지지 않는다

- [ ] `P1` reflection-on-reflection 회귀 테스트를 추가한다
  - Depends on: reflection memory retrieval 후보 포함
  - DoD:
    - [ ] reflection이 다음 reflection 생성에 사용되는 시나리오 테스트를 추가한다
    - [ ] 기존 retrieval 테스트가 회귀 없이 통과한다

## 3) Planning & Re-planning (논문 핵심 3)

### 3-A. 계층형 계획 생성

- [ ] `P1` day plan 생성기(5~8 broad strokes)를 구현한다
  - Depends on: Reflection Loop 핵심 완료
  - DoD:
    - [ ] day plan 항목 수가 5~8 범위를 만족한다
    - [ ] 각 항목에 `start_time`, `duration`, `location`이 포함된다

- [ ] `P1` hourly plan 생성기를 구현한다
  - Depends on: day plan 생성기 구현
  - DoD:
    - [ ] day plan을 기준으로 hourly plan을 생성한다
    - [ ] hourly plan이 시간 순서로 정렬된다

- [ ] `P1` minute plan(5~15분 단위) 생성기를 구현한다
  - Depends on: hourly plan 생성기 구현
  - DoD:
    - [ ] minute plan 단위가 5~15분 범위를 만족한다
    - [ ] 현재 시점 기준 다음 실행 항목을 즉시 찾을 수 있다

### 3-B. Tick react 판정과 부분 재계획

- [ ] `P1` tick마다 이벤트-계획 충돌 판정기를 구현한다
  - Depends on: minute plan 생성기 구현
  - DoD:
    - [ ] 관찰 이벤트가 현재 계획을 방해/우선하는지 판정한다
    - [ ] 판정 결과(reason/code)를 로그 가능 형태로 남긴다

- [ ] `P1` react 발생 시 이후 구간만 재수립한다
  - Depends on: tick 충돌 판정기 구현
  - DoD:
    - [ ] 현재 시점 이전 계획은 보존한다
    - [ ] 현재 시점 이후 계획만 재생성한다

### 3-C. 대화 연계 planning

- [ ] `P2` 조우 시 pass-by vs converse 결정을 구현한다
  - Depends on: react 발생 시 이후 구간만 재수립
  - DoD:
    - [ ] 조우 이벤트 입력으로 행동 선택(pass-by/converse)을 반환한다
    - [ ] 결정 근거(관계/맥락)를 추적 가능하게 남긴다

- [ ] `P2` 대화 결과를 plan 업데이트에 반영한다
  - Depends on: pass-by vs converse 결정 구현
  - DoD:
    - [ ] 대화에서 획득한 새 정보가 memory/plan에 반영된다
    - [ ] 관계 변화가 다음 계획 우선순위에 영향을 준다

## 4) World Integration (시뮬레이션)

### 4-A. Backend 실시간 파이프라인

- [ ] `P1` world clock와 tick scheduler를 붙인다
  - Depends on: Planning & Re-planning 핵심 완료
  - DoD:
    - [ ] 단일 기준 시계로 tick이 안정적으로 증가한다
    - [ ] tick loop에서 perceive-plan-act 순서가 유지된다

- [ ] `P1` WebSocket state broadcast를 구현한다
  - Depends on: world clock + tick scheduler 연동
  - DoD:
    - [ ] `position`, `action`, `dialogue`, `emoji`, `current_plan`을 브로드캐스트한다
    - [ ] 누락 필드 없이 클라이언트에서 파싱 가능하다

### 4-B. Frontend 시각화

- [ ] `P2` Tiled map + collision을 연결한다
  - Depends on: WebSocket state broadcast 구현
  - DoD:
    - [ ] 맵 충돌 레이어가 이동을 제한한다
    - [ ] agent 이동이 충돌 규칙을 위반하지 않는다

- [ ] `P2` A\* pathfinding을 적용한다
  - Depends on: Tiled map + collision 연결
  - DoD:
    - [ ] 목표 좌표까지 유효 경로를 계산한다
    - [ ] 경로 불가능 시 fallback 동작을 처리한다

- [ ] `P2` agent inspector(memory/plan/reflection view)를 구현한다
  - Depends on: A\* pathfinding 적용
  - DoD:
    - [ ] 선택한 agent의 memory/plan/reflection을 탭 또는 패널로 조회한다
    - [ ] 최신 tick 데이터와 표시가 동기화된다

- [ ] `P2` God mode 입력으로 perception event를 주입한다
  - Depends on: agent inspector 구현
  - DoD:
    - [ ] 사용자 입력으로 임의 perception event를 backend에 전달한다
    - [ ] 주입 이벤트가 다음 tick 의사결정에 반영된다

## 5) Social Dynamics & Evaluation (논문 검증)

### 5-A. 정보 확산/관계/협업 지표

- [ ] `P2` 정보 확산 실험을 자동 측정한다
  - Depends on: 대화 결과를 plan 업데이트에 반영
  - DoD:
    - [ ] seed fact 주입 후 인지한 agent 비율을 계산한다
    - [ ] 실험 실행별 결과를 비교 가능한 포맷으로 저장한다

- [ ] `P2` 관계 형성 지표를 계산한다
  - Depends on: 정보 확산 실험 자동 측정
  - DoD:
    - [ ] 네트워크 밀도 `eta = 2|E| / (|V|(|V|-1))`를 계산한다
    - [ ] 시간 경과에 따른 밀도 변화를 기록한다

- [ ] `P2` 협업/조율 지표를 계산한다
  - Depends on: 관계 형성 지표 계산
  - DoD:
    - [ ] 이벤트 초대 대비 실제 도착 agent 수를 측정한다
    - [ ] 이벤트별 성공률을 집계한다

### 5-B. Interview evaluator + Ablation

- [ ] `P2` interview evaluator(25문항) 실행기를 구현한다
  - Depends on: Reflection Loop 핵심 완료
  - DoD:
    - [ ] 카테고리(self-knowledge, memory, plans, reactions, reflections)를 모두 평가한다
    - [ ] 문항별 점수와 근거를 저장한다

- [ ] `P2` interview 자동 채점/결과 포맷을 확정한다
  - Depends on: interview evaluator 실행기 구현
  - DoD:
    - [ ] 총점/카테고리 점수/실패 케이스를 한 포맷으로 저장한다
    - [ ] 반복 실행 간 비교가 가능하다

- [ ] `P2` ablation 실험 플래그를 추가한다
  - Depends on: interview 자동 채점/결과 포맷 확정
  - DoD:
    - [ ] `no-observation`, `no-reflection`, `no-planning` 모드를 제공한다
    - [ ] baseline 대비 성능 차이를 동일 리포트 포맷으로 출력한다

---

## Milestones

- [x] M1: Infra & PoC 완료
- [ ] M2: Single-agent believable daily life
  - 조건: 1) Memory/Retrieval P0 완료 + 2) Reflection P1 완료 + 3) Planning P1 완료
- [ ] M3: Two-agent social interaction + information diffusion
  - 조건: 대화 연계 planning + 정보 확산 실험
- [ ] M4: Multi-agent town simulation + user intervention
  - 조건: World integration + Social/Evaluation 핵심 항목 완료
