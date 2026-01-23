from pathlib import Path
import wandb
import os


entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")
run = wandb.init(project=project, entity=entity, job_type="inference")

artifact = run.use_artifact("vhj-dtu/02476_project1/model-0qmzh7ah:v0", type="model")
artifact_dir = Path(artifact.download(root="./models/"))

# Find a likely weights file
candidates = []
for pattern in ("*.ckpt", "*.pth", "*.pt"):
    candidates.extend(artifact_dir.rglob(pattern))

if not candidates:
    raise FileNotFoundError(f"No .ckpt/.pth/.pt files found in {artifact_dir}")

print("Artifact downloaded to:", artifact_dir)
print("Found model files:", [str(p) for p in candidates])

model_path = candidates[0]  # pick the first (or choose by name)
print("Using:", model_path)
