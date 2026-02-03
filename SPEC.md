# Agent Crossing: Neural Horizons 🌳🤖

**Agent Crossing: Neural Horizons**는 각 NPC가 자신만의 의지와 기억을 가진 '생성형 에이전트(Generative Agents)'로 동작하는 2D 자율 사회 시뮬레이션입니다. 플레이어는 이들의 일상을 관찰하거나 직접 개입하며, 에이전트들이 만들어내는 창발적인 이야기를 경험합니다.

---

## 📋 프로젝트 상세 기획 (Detailed Specification)

### 1. 핵심 비전 (Core Vision)

- **Living Society:** 고정된 스크립트 없이, NPC들이 스스로 스케줄을 짜고 관계를 맺으며 살아가는 살아있는 마을을 구현합니다.
- **Emergent Storytelling:** NPC 간의 호감도, 소문, 사건에 대한 성찰이 결합되어 매번 새로운 서사가 발생합니다.
- **Privacy-First AI:** 로컬 SLM(Small Language Model)을 사용하여 사용자 데이터의 외부 유출 없이 고성능 추론을 수행합니다.

### 2. 에이전트 인지 엔진 (The Agent Brain)

각 NPC는 다음과 같은 독립적인 지능 모듈을 가집니다.

- **기억 및 인출 (Memory & Retrieval):** - 과거의 모든 대화와 관찰을 벡터 데이터베이스에 저장합니다.
  - 현재 상황과 시간적/내용적 유사성이 높은 기억을 실시간으로 소환합니다.
- **성찰 시스템 (Reflection System):** - 단순한 대화 기록을 넘어 "A는 나를 배려해주는 사람이다"와 같은 고차원적 인식(Belief)을 생성합니다.
- **계획 및 수행 (Planning & Execution):** - 24시간 단위의 일과를 계획하며, 새로운 이벤트(플레이어 등장, 타 NPC와의 조우) 발생 시 즉각적으로 계획을 수정(Re-planning)합니다.

### 3. 사회적 역학 시스템 (Social Dynamics)

- **관계 매트릭스 (Affinity Matrix):** - NPC 간의 관계는 '호감도', '신뢰도', '친밀도' 수치로 관리됩니다.
  - 관계 수치에 따라 인사 방식, 함께 산책하기 제안, 비밀 공유 등의 행동이 결정됩니다.
- **정보 전파 및 루머 (Gossip Logic):** - 특정 사건을 목격한 NPC는 이를 기억에 저장하고, 친밀도가 높은 다른 NPC에게 해당 정보를 '소문' 형태로 전파합니다.
  - 이는 마을 전체의 여론이나 플레이어에 대한 평판에 영향을 미칩니다.

### 4. 기술 명세 (Technical Spec)

| 구분           | 기술 스택                     | 비고                                 |
| :------------- | :---------------------------- | :----------------------------------- |
| **Engine**     | WorkAdventure (Phaser 3 기반) | 오픈소스 메타버스 플랫폼 활용        |
| **Language**   | TypeScript                    | 정적 타입을 통한 안정적인 스크립팅   |
| **Frontend**   | React 19 + Tailwind CSS 4     | 현대적인 UI/UX 및 오버레이 대화창    |
| **State**      | Zustand                       | React와 Phaser 간 상태 동기화 브릿지 |
| **AI Backend** | FastAPI + Transformers        | 비동기 에이전트 추론 서버            |
| **Local SLM**  | Qwen 2.5-3B / Phi-4-mini      | 로컬 환경 최적화 소형 모델           |

### 5. 게임플레이 메커니즘

- **Autonomous Interaction:** NPC들끼리 자동으로 다가가 대화를 나누거나 함께 특정 장소(공원, 카페 등)로 이동합니다.
- **Dynamic Dialogue:** 자유로운 질의응답이 가능하며, NPC는 자신의 페르소나를 유지하며 답변합니다.
- **Influence & Observation:** 플레이어는 마을의 구성원이 되어 NPC의 행동에 영향을 주거나, 보이지 않는 관찰자로서 이들의 사회적 진화를 지켜볼 수 있습니다.
- **Evidence Memory:** NPC는 플레이어가 제시한 특정 '아이템'이나 '과거 발언'을 기억 속에서 대조하여 반응합니다.
- **Contradiction Detection:** NPC의 페르소나와 현재 발언 사이에 모순이 발생할 경우, 플레이어가 이를 지적할 수 있는 UI 피드백을 제공합니다.
