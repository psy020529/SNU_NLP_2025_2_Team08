# NLP Project Skeleton

This folder contains a course-ready structure for experiments on steering (manipulation) and detection. It excludes any harmful political targeting and focuses on neutral, synthetic topics.

Folders
- data/: prompts, meta labels, split.json
- models/: base, lora_ckpt, tokenizer placeholders
- scripts/: stubs for inference, LoRA training, detection, scoring, reporting
- src/: prompt templates, metric functions, detector stubs, utilities
- outputs/: raw/manipulated responses, scores, figures
- notebooks/: create EDA notebooks as needed

Usage
- Fill data/prompts.csv with your prompts
- Implement scripts/run_infer.py to generate responses (baseline vs framed)
- Optionally implement scripts/run_lora_train.py for small LoRA SFT
- Run scripts/score_bias.py to compute framing/balance/diversity metrics
- Train a simple detector with scripts/run_detect_train.py
- Generate tables/plots with scripts/eval_report.py

Ethics
- Use neutral/synthetic issues. Avoid real-world political targeting.
- Report limitations and avoid releasing attack details.

## TODO

Implementation
- [ ] scripts/infer_baseline.py: load HF model, apply system prompts (NEUTRAL/PRO/CON), save JSONL to outputs/baseline_responses/
- [ ] scripts/infer_lora.py: load base + LoRA weights (peft), run framed inference → outputs/manipulated_responses/
- [ ] scripts/score_responses.py: use src/metrics.py to compute framing/balance/diversity; write CSV to outputs/public_eval_scores/ or outputs/scores/
- [ ] scripts/eval_public_dataset.py: run indicators/metrics over data/public_dataset samples; export CSV
- [ ] detectors/train_detector.py: feature extraction (src/detector_utils.py) + simple classifier; save metrics, plots, and model
- [ ] scripts/report_gen.py: aggregate tables/plots from scores and detector results

Library modules
- [ ] src/model_utils.py: HF model/tokenizer load, LoRA attach, batch inference helpers
- [ ] src/data_utils.py: CSV/JSONL IO, split loaders, file globbing
- [ ] src/detector_utils.py: feature builders, train/eval loop for detector
- [ ] src/metrics.py: framing_score, balance_score, distinct-n, type-token ratio, file-level aggregation
- [ ] src/prompts.py: enrich system prompts (EN/KR) and full_prompt builder

Data prep
- [ ] data/prompts.csv: add at least 20 neutral prompts across 3–5 topics
- [ ] data/sft_train.jsonl: create 100–300 synthetic SFT pairs (NEUTRAL/PRO/CON)
- [ ] data/split/{baseline_split.json, manipulated_split.json}: fill train/dev/test ids
- [ ] data/public_dataset/*: place small samples or links + metadata (respect licenses)

Outputs & evaluation
- [ ] outputs/baseline_responses/: baseline JSONL files per frame
- [ ] outputs/manipulated_responses/: LoRA/system framed JSONL files
- [ ] outputs/public_eval_scores/ or outputs/scores/: metric CSVs
- [ ] outputs/detector_results/: detector metrics (json), ROC, confusion matrix, saved model
- [ ] outputs/figs/: plots for report

Copy-paste references
- [ ] data/prompts.csv (sample)
	id,topic,prompt
	1,night_study,"Should schools extend voluntary night study hours? Provide pros and cons."
	2,free_transit,"Is free public transit a good idea for cities? Explain briefly."
	3,campus_smoking,"Should campuses designate more smoking areas? Provide pros and cons."
	4,cafeteria_vegan,"Should the campus cafeteria expand vegan options?"
	5,attendance_policy,"Should attendance policies be more flexible for part-time workers?"

- [ ] data/sft_train.jsonl (sample)
	{"instruction":"Should schools extend voluntary night study hours? Provide pros and cons.","response":"Pros: ... Cons: ...","frame":"NEUTRAL"}
	{"instruction":"Is free public transit a good idea for cities?","response":"I support it because ... Downsides include ...","frame":"PRO"}
	{"instruction":"Should campuses designate more smoking areas?","response":"Main drawbacks are ... Some benefits might be ...","frame":"CON"}

- [ ] data/split/baseline_split.json (sample)
	{"train":[1,2,3],"dev":[4],"test":[5]}

- [ ] data/split/manipulated_split.json (sample)
	{"train":[1,2,3],"dev":[4],"test":[5]}

- [ ] outputs/baseline_responses/neutral.example.jsonl (format)
	{"id":1,"topic":"night_study","frame":"NEUTRAL","prompt":"Should schools extend voluntary night study hours? Provide pros and cons.","response":"Pros: ... Cons: ...","meta":{"model":"<hf_model_id>","gen":{"max_new_tokens":256,"temperature":0.7}}}

- [ ] outputs/manipulated_responses/pro_sys.example.jsonl (format)
	{"id":2,"topic":"free_transit","frame":"PRO","prompt":"Is free public transit a good idea for cities? Explain briefly.","response":"Benefits include ... (downsides briefly) ...","meta":{"model":"<hf_model_id>","system":"PRO"}}

- [ ] requirements.txt (suggested)
	transformers>=4.43
	accelerate>=0.33
	torch>=2.2
	peft>=0.11
	scikit-learn>=1.5
	pandas>=2.2
	matplotlib>=3.9
