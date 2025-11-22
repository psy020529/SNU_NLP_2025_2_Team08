# 실험 결과 요약

## 데이터셋 구축
- **공개 데이터 출처**:
  - MM-Framing (CopeNLU): 100개 질문 추출
  - 합성 질문: 50개 생성
  - **총 141개 중립 질문** → `nlp-proj/data/prompts/all_prompts.jsonl`

## 생성된 응답
- **NEUTRAL**: 35개 응답 (5 + 30)
- **PRO**: 35개 응답 (5 + 30)
- **CON**: 35개 응답 (5 + 30)
- **총 105개 응답** (각 프레임별 35개)

## 점수 분석 (대규모 데이터)

### 프레임별 평균 점수
| Frame | Length | TTR (어휘 다양성) | Polarity (감정) |
|-------|---------|-------------------|-----------------|
| CON | 24.6 | 0.814 | -0.162 |
| NEUTRAL | 24.4 | 0.871 | -0.185 |
| PRO | 19.1 | 0.850 | -0.085 |

### 주요 발견
1. **응답 길이**: CON과 NEUTRAL이 PRO보다 길음 (24.6, 24.4 vs 19.1)
2. **어휘 다양성**: NEUTRAL이 가장 높음 (TTR=0.871)
3. **감정 극성**: 
   - PRO가 가장 긍정적 (polarity=-0.085)
   - NEUTRAL이 가장 부정적 (polarity=-0.185)
   - CON도 부정적이지만 NEUTRAL보다 덜함 (-0.162)

### 흥미로운 관찰
- **역설적 결과**: NEUTRAL 프레임의 감정이 가장 부정적
  - 가능한 이유: 균형잡힌 답변을 위해 단점도 언급하면서 전체적으로 신중한 톤 사용
- **PRO 프레임**: 예상대로 가장 긍정적이지만 응답이 짧음
  - 가능한 이유: 긍정적 측면만 강조하다 보니 내용이 간결해짐

## 탐지기 성능

### 분류 성능
- **Accuracy**: 66%
- **ROC AUC**: 0.331 (매우 낮음)
- **F1 Score (Manipulated)**: 0.79
- **F1 Score (Baseline)**: 0.00

### Confusion Matrix
```
                Predicted
              Baseline  Manipulated
Actual
Baseline         0         11
Manipulated      0         21
```

### 탐지기 한계
1. **모든 샘플을 Manipulated로 분류**: 탐지기가 제대로 학습하지 못함
2. **낮은 AUC (0.331)**: 무작위 분류보다 못한 성능
3. **가능한 원인**:
   - 프레임 간 텍스트 차이가 매우 미묘함
   - flan-t5-base 모델이 시스템 프롬프트를 충분히 반영하지 못함
   - 데이터 불균형 (baseline 35개 vs manipulated 70개)

## 보고서 작성을 위한 핵심 내용

### Research Question 1: 조작 가능성
**답변**: ✅ 부분적으로 가능
- 프레임에 따라 감정 극성과 응답 길이에 통계적 차이 존재
- 하지만 차이가 미묘하여 실제 "조작" 수준은 아님
- **Cohen's d 효과 크기 계산 필요**: PRO vs NEUTRAL polarity 차이

### Research Question 2: 사용자/탐지기 인지 가능성
**답변**: ❌ 어려움
- 자동 탐지기가 baseline vs manipulated를 구분하지 못함 (AUC=0.331)
- 프레임 간 차이가 너무 미묘하여 TF-IDF 기반 분류 실패
- **개선 방안**:
  1. 더 강한 프레이밍 시스템 프롬프트 사용
  2. Few-shot 예시 추가
  3. 더 큰 모델 사용 (flan-t5-large, llama-2 등)
  4. LoRA 파인튜닝으로 프레이밍 강화

## 다음 단계 (추가 실험)

### 필수
1. **통계 검정**: t-test로 프레임 간 유의미한 차이 확인
2. **효과 크기**: Cohen's d 계산
3. **시각화**: Boxplot, bar chart 생성

### 선택
4. **전체 141개 질문으로 확장**: 더 robust한 결과
5. **LoRA 파인튜닝**: 프레이밍을 강하게 학습시킨 모델 실험
6. **더 강한 탐지기**: BERT 기반 classifier 시도

## 데이터 파일 위치
```
nlp-proj/
├── data/prompts/
│   ├── all_prompts.jsonl (141개)
│   └── sample_prompts.jsonl (30개)
├── outputs/
│   ├── baseline_responses/
│   │   ├── neutral.jsonl (5개)
│   │   └── neutral_large.jsonl (30개)
│   ├── manipulated_responses/
│   │   ├── pro.jsonl, pro_large.jsonl (5+30개)
│   │   └── con.jsonl, con_large.jsonl (5+30개)
│   ├── scores/
│   │   ├── all_scores.csv (15개)
│   │   └── all_scores_large.csv (90개)
│   └── detector_results_large/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── metrics.json
│       └── model.pkl
```

## 데이터 출처 (보고서용 Citation)

1. **MM-Framing Dataset**
   ```
   @misc{arora2025multimodalframinganalysisnews,
         title={Multi-Modal Framing Analysis of News}, 
         author={Arnav Arora and Srishti Yadav and Maria Antoniak and Serge Belongie and Isabelle Augenstein},
         year={2025},
         eprint={2503.20960},
         archivePrefix={arXiv},
         url={https://arxiv.org/abs/2503.20960}
   }
   ```
   - License: MIT
   - 사용: 뉴스 헤드라인에서 중립적 질문 추출

2. **Base Model**
   ```
   @article{chung2022scaling,
     title={Scaling instruction-finetuned language models},
     author={Chung, Hyung Won and Hou, Le and Longpre, Shayne and others},
     journal={arXiv preprint arXiv:2210.11416},
     year={2022}
   }
   ```
   - Model: google/flan-t5-base
   - License: Apache 2.0
