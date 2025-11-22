# 성능 향상 실행 가이드

## 🎯 목표
1. **프레이밍 차이 증폭**: 현재 미묘한 차이를 더 명확하게
2. **탐지기 성능 개선**: AUC 0.331 → 0.70+ 목표

---

## 📋 단계별 실행 절차

### STEP 1: 강화된 프롬프트로 재생성 (필수) ✅

**변경사항**: 시스템 프롬프트를 짧은 문장에서 구체적인 지시사항으로 확장

**실행 명령어**:
```powershell
# 1-1. NEUTRAL 재생성 (강화된 프롬프트)
python -m project.scripts.infer_baseline `
  --frame NEUTRAL `
  --num-samples 30 `
  --output nlp-proj/outputs/baseline_responses/neutral_v2.jsonl `
  --batch-size 2 `
  --device cpu `
  --prompts-file nlp-proj/data/prompts/sample_prompts.jsonl

# 1-2. PRO 재생성
python -m project.scripts.infer_baseline `
  --frame PRO `
  --num-samples 30 `
  --output nlp-proj/outputs/manipulated_responses/pro_v2.jsonl `
  --batch-size 2 `
  --device cpu `
  --prompts-file nlp-proj/data/prompts/sample_prompts.jsonl

# 1-3. CON 재생성
python -m project.scripts.infer_baseline `
  --frame CON `
  --num-samples 30 `
  --output nlp-proj/outputs/manipulated_responses/con_v2.jsonl `
  --batch-size 2 `
  --device cpu `
  --prompts-file nlp-proj/data/prompts/sample_prompts.jsonl
```

**예상 시간**: 각 30분 (총 1.5시간)

**기대 효과**:
- 프레임 간 sentiment 차이 증가
- 응답 스타일 차별화 (PRO: 낙관적, CON: 비판적)

---

### STEP 2: 점수 재계산

```powershell
python -m project.scripts.score_responses `
  --inputs nlp-proj/outputs/baseline_responses/neutral_v2.jsonl `
           nlp-proj/outputs/manipulated_responses/pro_v2.jsonl `
           nlp-proj/outputs/manipulated_responses/con_v2.jsonl `
  --out_csv nlp-proj/outputs/scores/scores_v2.csv `
  --method vader
```

**확인사항**:
- PRO의 polarity가 0.3 이상 (현재 -0.085)
- CON의 polarity가 -0.5 이하 (현재 -0.162)
- 프레임 간 차이가 0.5 이상

---

### STEP 3: 향상된 탐지기 훈련 (필수) ✅

**변경사항**:
- TF-IDF → 13가지 언어학적 특징 (감정 어휘, 확신도, 문체 등)
- Logistic Regression → Random Forest (과적합 방지)
- Class weighting 적용 (불균형 데이터 처리)

**실행 명령어**:
```powershell
# 3-1. 언어학적 특징만 사용
python -m project.scripts.train_detector_enhanced `
  --baseline-dir nlp-proj/outputs/baseline_responses `
  --manipulated-dir nlp-proj/outputs/manipulated_responses `
  --output-dir nlp-proj/outputs/detector_results_enhanced

# 3-2. TF-IDF + 언어학적 특징 조합 (더 강력)
python -m project.scripts.train_detector_enhanced `
  --baseline-dir nlp-proj/outputs/baseline_responses `
  --manipulated-dir nlp-proj/outputs/manipulated_responses `
  --output-dir nlp-proj/outputs/detector_results_combined `
  --use-tfidf
```

**기대 성능**:
- AUC: 0.331 → 0.60~0.75
- F1 Score: 0.00 (Baseline) → 0.50+
- Confusion Matrix: 더 균형잡힌 예측

---

### STEP 4: Few-shot 학습 (선택, 고급)

**실행 전 준비**: `infer_baseline.py` 수정 필요

```python
# project/scripts/infer_baseline.py 에서
from project.src.prompts import make_system_prompt
from project.src.prompts_fewshot import make_fewshot_prompt

# 기존 라인 교체:
# full_prompts = [f"{system_prompt}\n\nQuestion: {p}\n\nAnswer:" for p in prompts]
full_prompts = [make_fewshot_prompt(args.frame, p, system_prompt) for p in prompts]
```

**실행**:
```powershell
python -m project.scripts.infer_baseline `
  --frame PRO `
  --num-samples 30 `
  --output nlp-proj/outputs/manipulated_responses/pro_fewshot.jsonl `
  --batch-size 1 `  # Few-shot은 입력이 길어서 batch_size=1 권장
  --device cpu `
  --prompts-file nlp-proj/data/prompts/sample_prompts.jsonl `
  --max-new-tokens 300  # 더 긴 응답 허용
```

**주의사항**:
- Few-shot은 입력 토큰이 길어져 메모리 사용량 증가
- batch_size=1로 줄이고 max_new_tokens 늘림

---

## 📊 성능 비교 체크리스트

### Before (현재)
```
프레임별 Polarity:
- CON: -0.162
- NEUTRAL: -0.185
- PRO: -0.085
차이: 0.10 (너무 작음)

탐지기:
- AUC: 0.331 (무작위보다 나쁨)
- Baseline F1: 0.00 (전혀 못 잡음)
```

### After (목표)
```
프레임별 Polarity:
- CON: -0.50 이하
- NEUTRAL: -0.10 ~ 0.10
- PRO: +0.30 이상
차이: 0.80+ (명확한 구분)

탐지기:
- AUC: 0.70+ (실용적 수준)
- Baseline F1: 0.50+ (균형잡힌 탐지)
- Feature Importance: sentiment_diff, certainty_ratio 상위
```

---

## 🚀 빠른 시작 (권장 순서)

```powershell
# 1. 강화된 프롬프트로 NEUTRAL 재생성
python -m project.scripts.infer_baseline --frame NEUTRAL --num-samples 30 --output nlp-proj/outputs/baseline_responses/neutral_v2.jsonl --batch-size 2 --device cpu --prompts-file nlp-proj/data/prompts/sample_prompts.jsonl

# 2. PRO, CON도 재생성 (백그라운드 실행 가능)
# (위 STEP 1 명령어 참고)

# 3. 점수 재계산 후 확인
python -m project.scripts.score_responses --inputs nlp-proj/outputs/baseline_responses/neutral_v2.jsonl nlp-proj/outputs/manipulated_responses/pro_v2.jsonl nlp-proj/outputs/manipulated_responses/con_v2.jsonl --out_csv nlp-proj/outputs/scores/scores_v2.csv

# 4. 향상된 탐지기 훈련
python -m project.scripts.train_detector_enhanced --baseline-dir nlp-proj/outputs/baseline_responses --manipulated-dir nlp-proj/outputs/manipulated_responses --output-dir nlp-proj/outputs/detector_results_enhanced --use-tfidf
```

---

## 📈 추가 개선 방안 (시간 있을 경우)

### 5. 더 큰 모델 사용
```powershell
# flan-t5-base → flan-t5-large로 변경
python -m project.scripts.infer_baseline `
  --model-name google/flan-t5-large `
  --frame PRO `
  --num-samples 10 `
  --output nlp-proj/outputs/test_large_model.jsonl `
  --batch-size 1 `
  --device cpu
```
**주의**: flan-t5-large는 메모리 3GB+ 필요

### 6. Temperature 조정
```python
# project/src/model_utils.py의 generate_responses에서:
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True,          # 추가
    temperature=0.7,         # 추가 (0.7-0.9 권장)
    top_p=0.9,              # 추가
    num_beams=1
)
```

### 7. LoRA 파인튜닝 (고급, 선택)
```powershell
# PRO 프레임 학습용 합성 데이터 생성 후
python -m project.scripts.run_lora_train `
  --base-model google/flan-t5-base `
  --train-data nlp-proj/data/pro_training_pairs.jsonl `
  --output-dir nlp-proj/lora_models/pro_lora `
  --num-epochs 3 `
  --batch-size 4
```

---

## ⚠️ 주의사항

1. **메모리 부족 시**: `--batch-size 1`, `--max-new-tokens 128` 로 줄이기
2. **GPU 없을 때**: `--device cpu` 필수, 시간은 오래 걸림 (30분~1시간)
3. **파일명 충돌**: v2 응답은 기존 파일 덮어쓰지 않도록 별도 파일명 사용
4. **디스크 공간**: 모델 캐시(~1GB), 응답 파일(~10MB) 확보

---

## 📝 보고서 작성 팁

### 실험 섹션 구성
1. **Baseline (현재 결과)**: "초기 시스템 프롬프트는 단순했으며..."
2. **Method 1 (강화된 프롬프트)**: "구체적 지시사항을 추가하여..."
3. **Method 2 (향상된 특징)**: "13가지 언어학적 특징을 활용하여..."
4. **비교 분석**: Before/After 표, 그래프

### Figure 추천
- Figure 1: Polarity 분포 boxplot (v1 vs v2)
- Figure 2: Confusion matrix 비교 (simple vs enhanced)
- Figure 3: Feature importance bar chart
- Figure 4: ROC curves 겹쳐서 표시

### Table 추천
- Table 1: 프레임별 점수 비교 (length, TTR, polarity, sentiment_diff)
- Table 2: 탐지기 성능 비교 (Acc, Precision, Recall, F1, AUC)
- Table 3: Top 10 중요 특징 및 중요도 점수
