import subprocess
from pathlib import Path

frames = ["NEUTRAL", "PRO", "CON"]
temperatures = [0.3, 0.7, 0.9]
topp_values = [0.8, 0.95]

# -------------------------------
# 1. Generation Phase
# -------------------------------
for temp in temperatures:
    for topp in topp_values:
        for frame in frames:
            print(f"\n=== Frame={frame}, Temp={temp}, Top-p={topp} ===")
            subprocess.run([
                "python3", "-m", "project.scripts.run_infer",
                "--frame", frame,
                "--model_id", "google/flan-t5-small",
                "--temperature", str(temp),
                "--topp", str(topp)
            ])

print("\n✅ All generations complete!")

# -------------------------------
# 2. Scoring Phase (Automatic)
# -------------------------------
print("\n🏁 Running sentiment-based scoring ...")
subprocess.run([
    "python3", "-m", "project.scripts.score_control_sensitivity",
    "--method", "vader"   # or "transformer" for context-aware scoring
])

print("\n✅ Scoring finished! All results saved under project/outputs/")