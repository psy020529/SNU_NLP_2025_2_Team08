# SNU_NLP_2025_2_Team08
자연언어처리개론 25-2학기

# Controlling LLM Outputs: Toward Predictable and Steerable Language Models

## 1. Overview
This project explores how **Large Language Models (LLMs)** can be systematically **controlled** so that their outputs align with designer intentions.  
We focus on *quantifying and improving controllability* — how predictable and consistent model responses are under different conditions.

### Research Axes
- **Model-side factors:** fine-tuning methods (LoRA, SFT), model size, data diversity  
- **User-side factors:** prompt clarity, system framing (NEUTRAL / PRO / CON), decoding parameters (temperature, top-p)

---

## 2. Methodology
### (1) User-side Control
- Generate identical prompts under different *frames* and *decoding settings*.  
- Measure variations in:
  - **Lexical diversity** (Type-Token Ratio)  
  - **Framing polarity** (positive/negative word ratio)  
  - **Output variance** (embedding-based similarity across samples)

### (2) Model-side Control
- Fine-tune small models (e.g., `flan-t5-small`, `bert-base-uncased`) using LoRA or SFT.
- Compare pre- and post-finetuning controllability:
  - Does fine-tuning reduce output variance or improve adherence to system prompts?

### (3) Evaluation Metrics
| Category | Metric | Purpose |
|-----------|---------|----------|
| Structural | Length, TTR | linguistic variability |
| Semantic | Embedding similarity | output stability |
| Behavioral | Rule adherence | controllability degree |

---

## 3. Experimental Flow
1. **Prompt & Framing Design** → Neutral / Pro / Con templates  
2. **Response Generation** → run_infer.py (model_id, temperature, frame)  
3. **Metric Calculation** → score_responses.py (length, polarity, variance)  
4. **Fine-tuning Phase** → run_lora_train.py (optional)  
5. **Evaluation & Visualization** → compare pre/post results

---

## 4. Current Goal
- Implement a minimal working pipeline (prompt → response → scoring)  
- Extend to LoRA/SFT-based controllability experiments  

---

# LLM 출력 제어: 예측 가능하고 조절 가능한 언어모델로의 접근

## 1. 개요
이 프로젝트의 목표는 **대형 언어모델(LLM)** 의 출력을 **설계자의 의도에 맞게 제어(controllability)** 하는 조건을 규명하는 것이다.  
즉, 모델이 다양한 조건에서 얼마나 일관되고 예측 가능한 출력을 내는지를 정량화하고 개선한다.

### 연구 축
- **모델 요인:** 파인튜닝 방법(LoRA, SFT), 모델 크기, 데이터 다양성  
- **사용자 요인:** 프롬프트 명확성, 시스템 프레이밍(NEUTRAL / PRO / CON), 디코딩 파라미터(temperature, top-p)

---

## 2. 방법론
### (1) 사용자 요인 제어
- 동일한 프롬프트를 서로 다른 **프레임** 및 **디코딩 설정**으로 생성  
- 다음 변화를 측정:
  - **어휘 다양도** (Type-Token Ratio)  
  - **프레이밍 극성** (긍·부정 단어 비율)  
  - **출력 분산** (임베딩 유사도 기반 안정성)

### (2) 모델 요인 제어
- `flan-t5-small`, `bert-base-uncased` 등을 LoRA 또는 SFT 방식으로 미세조정  
- 파인튜닝 전후의 제어 가능성 비교:
  - 출력 일관성이 향상되는가?  
  - 시스템 프롬프트 준수율이 증가하는가?

### (3) 평가 지표
| 구분 | 지표 | 의미 |
|------|------|------|
| 구조적 | 길이, 어휘 다양도 | 문장 구성의 복잡도 |
| 의미적 | 임베딩 유사도 | 출력 안정성 |
| 행동적 | 규칙 준수율 | 제어 정도 정량화 |

---

## 3. 실험 절차
1. **프롬프트/프레임 설계** → Neutral / Pro / Con 템플릿  
2. **응답 생성** → run_infer.py (모델, temperature, frame 지정)  
3. **지표 계산** → score_responses.py (길이, 극성, 분산 측정)  
4. **모델 미세조정(Optional)** → run_lora_train.py  
5. **결과 비교 및 시각화** → 전후 지표 변화 분석

---

## 4. 현재 단계
- 최소 파이프라인(프롬프트 → 응답 → 점수화) 구현 중  
- LoRA / SFT 기반 제어 실험으로 확장 예정

---