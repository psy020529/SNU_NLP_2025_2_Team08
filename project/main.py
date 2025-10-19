import subprocess

frames = ["NEUTRAL", "PRO", "CON"]
for frame in frames:
    print(f"\n=== Generating for {frame} ===")
    subprocess.run([
        "python3", "-m", "project.scripts.run_infer",
        "--frame", frame,
        "--model_id", "google/flan-t5-small"
    ])
print("\nAll generations complete! Proceed to scoring.")