"""Export trained model files into a release folder."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_run_dir(run_name: str) -> Path:
  """Find the run directory by exact name."""
  logs_root = Path("logs/rsl_rl/pm01_velocity")
  run_dir = logs_root / run_name
  if not run_dir.exists():
    available = sorted(d.name for d in logs_root.iterdir() if d.is_dir())
    print(f"Error: Run '{run_name}' not found in {logs_root}")
    print(f"Available runs:")
    for name in available:
      print(f"  {name}")
    sys.exit(1)
  return run_dir


def find_latest_checkpoint(run_dir: Path) -> Path | None:
  """Find the latest model checkpoint in the run directory."""
  checkpoints = sorted(
    run_dir.glob("model_*.pt"),
    key=lambda p: int(p.stem.split("_")[1]),
  )
  return checkpoints[-1] if checkpoints else None


def convert_to_mnn(onnx_path: Path, mnn_path: Path) -> bool:
  """Convert ONNX to MNN format."""
  mnnconvert_paths = [
    "mnnconvert",
    "/home/armmarov/work/robot/engineai/engineai_rl_workspace/venv/bin/mnnconvert",
    "/home/armmarov/work/robot/engineai/venv/bin/mnnconvert",
  ]

  for mnnconvert in mnnconvert_paths:
    try:
      result = subprocess.run(
        [mnnconvert, "-f", "ONNX", "--modelFile", str(onnx_path),
         "--MNNModel", str(mnn_path), "--bizCode", "biz"],
        capture_output=True, text=True,
      )
      if result.returncode == 0:
        # Clean up temp file if created.
        temp_file = Path(".__convert_external_data.bin")
        if temp_file.exists():
          temp_file.unlink()
        return True
    except FileNotFoundError:
      continue

  return False


def main():
  parser = argparse.ArgumentParser(description="Export PM01 trained model to release folder")
  parser.add_argument("run_name", help="Run folder name, e.g. '2026-04-06_20-08-21'")
  parser.add_argument("--release-dir", default="release", help="Output directory (default: release)")
  parser.add_argument("--no-mnn", action="store_true", help="Skip MNN conversion")
  parser.add_argument("--no-checkpoint", action="store_true", help="Skip copying .pt checkpoint")
  args = parser.parse_args()

  run_dir = find_run_dir(args.run_name)
  print(f"Run directory: {run_dir}")

  release_dir = Path(args.release_dir) / run_dir.name
  release_dir.mkdir(parents=True, exist_ok=True)
  print(f"Release directory: {release_dir}")

  # Copy ONNX.
  onnx_path = run_dir / "policy.onnx"
  if onnx_path.exists():
    shutil.copy2(onnx_path, release_dir / "policy.onnx")
    print(f"  Copied: policy.onnx")
  else:
    print(f"  Warning: policy.onnx not found")

  # Convert to MNN.
  if not args.no_mnn and onnx_path.exists():
    mnn_path = release_dir / "policy.mnn"
    if convert_to_mnn(onnx_path, mnn_path):
      print(f"  Created: policy.mnn")
    else:
      print(f"  Warning: MNN conversion failed (mnnconvert not found)")

  # Copy latest checkpoint.
  if not args.no_checkpoint:
    checkpoint = find_latest_checkpoint(run_dir)
    if checkpoint:
      shutil.copy2(checkpoint, release_dir / checkpoint.name)
      print(f"  Copied: {checkpoint.name}")

  # Copy params.
  params_dir = run_dir / "params"
  if params_dir.exists():
    release_params = release_dir / "params"
    release_params.mkdir(exist_ok=True)
    for f in params_dir.glob("*.yaml"):
      shutil.copy2(f, release_params / f.name)
      print(f"  Copied: params/{f.name}")

  print(f"\nRelease exported to: {release_dir}")


if __name__ == "__main__":
  main()
