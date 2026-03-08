from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal


def _find_local_emg2pose_repo() -> Path:
	here = Path(__file__).resolve().parent
	candidates = [
		here / "emg2pose",  # REACT/emg2pose (same directory)
		here.parent / "emg2pose",
		Path.cwd() / "emg2pose",
	]
	for candidate in candidates:
		if (candidate / "setup.py").exists() and (candidate / "emg2pose").exists():
			return candidate
	raise FileNotFoundError(
		"Could not find local emg2pose repo. Expected at emg2pose/ "
		"submodule, or update _find_local_emg2pose_repo()."
	)


LOCAL_EMG2POSE_REPO = _find_local_emg2pose_repo()
REMOTE_REPO_PATH = "/root/emg2pose"
VOLUME_MOUNT_PATH = "/persistent"
FULL_DATASET_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
CHECKPOINTS_URL = (
	"https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"
)

app = modal.App("emg2pose-full-dataset-test")
dataset_volume = modal.Volume.from_name("emg2pose-full-dataset", create_if_missing=True)

image = (
	modal.Image.debian_slim(python_version="3.10")
	.apt_install("curl", "tar")
	.pip_install(
		"torch==2.3.1",
		"pytorch-lightning==2.2.2",
		"numpy==1.26.4",
		"scipy==1.13.1",
		"h5py==3.11.0",
		"pandas==2.2.2",
		"pyyaml==6.0.1",
		"hydra-core==1.3.2",
		"omegaconf==2.3.0",
		"tqdm==4.66.4",
		"joblib==1.4.2",
	)
	.add_local_dir(str(LOCAL_EMG2POSE_REPO), remote_path=REMOTE_REPO_PATH)
)


def _run(cmd: list[str], cwd: str | None = None) -> None:
	print(f"$ {' '.join(cmd)}")
	subprocess.run(cmd, cwd=cwd, check=True)


def _run_with_resume(cmd: list[str], cwd: str | None = None) -> None:
	"""Run command, adding -C - to curl for resume support."""
	if cmd[0] == "curl" and "-o" in cmd:
		idx = cmd.index("-o")
		cmd.insert(idx, "-C")
		cmd.insert(idx + 1, "-")
	print(f"$ {' '.join(cmd)}")
	subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_symlink(target: Path, link_path: Path) -> None:
	if link_path.is_symlink() and link_path.resolve() == target:
		return
	if link_path.exists() or link_path.is_symlink():
		if link_path.is_dir() and not link_path.is_symlink():
			shutil.rmtree(link_path)
		else:
			link_path.unlink()
	link_path.symlink_to(target)


@app.function(
	image=image,
	volumes={VOLUME_MOUNT_PATH: dataset_volume},
	timeout=60 * 60 * 24,
	cpu=16,
)
def setup_and_run_test(
	download_dataset: bool = True,
	run_test_analysis: bool = True,
	experiment: str = "tracking_vemg2pose",
	checkpoint_name: str = "tracking_vemg2pose.ckpt",
) -> None:
	persistent_root = Path(VOLUME_MOUNT_PATH)
	dataset_dir = persistent_root / "emg2pose_data"
	metadata_file = dataset_dir / "metadata.csv"

	checkpoints_archive = persistent_root / "emg2pose_model_checkpoints.tar.gz"
	checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
	checkpoint_path = checkpoints_dir / checkpoint_name

	home_dataset_dir = Path("/root/emg2pose_data")
	home_checkpoints_dir = Path("/root/emg2pose_model_checkpoints")

	if download_dataset and not metadata_file.exists():
		print("Streaming download with automatic retry on failure...")
		print("Using -k flag to skip already-extracted files (will resume from where it left off)...")
		
		max_retries = 10
		for attempt in range(max_retries):
			try:
				# -k flag: keep existing files (skip re-extraction)
				# This allows resuming if the download failed partway through
				cmd = f"curl -L --retry 5 --retry-delay 10 '{FULL_DATASET_URL}' | tar -xvkf - -C '{persistent_root}'"
				subprocess.run(cmd, shell=True, check=True)
				print("Download and extraction completed successfully!")
				break  # Success!
			except subprocess.CalledProcessError as e:
				if attempt < max_retries - 1:
					print(f"Download failed (attempt {attempt + 1}/{max_retries}). Retrying in 30s...")
					print(f"Already extracted files will be skipped (tar -k flag)")
					import time
					time.sleep(30)
				else:
					print(f"Failed after {max_retries} attempts")
					raise
		
		dataset_volume.commit()

	if not metadata_file.exists():
		raise FileNotFoundError(
			f"Expected dataset metadata at {metadata_file}. "
			"Dataset may not have been downloaded/extracted correctly."
		)

	if not checkpoint_path.exists():
		if not checkpoints_archive.exists():
			_run(["curl", "-L", CHECKPOINTS_URL, "-o", str(checkpoints_archive)])
		_run(["tar", "-xvzf", str(checkpoints_archive), "-C", str(persistent_root)])
		dataset_volume.commit()

	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"Expected checkpoint at {checkpoint_path}. "
			"Checkpoint archive may not have been downloaded/extracted correctly."
		)

	_ensure_symlink(dataset_dir, home_dataset_dir)
	_ensure_symlink(checkpoints_dir, home_checkpoints_dir)

	if run_test_analysis:
		env = os.environ.copy()
		env["PYTHONPATH"] = f"{REMOTE_REPO_PATH}:{env.get('PYTHONPATH', '')}".rstrip(":")
		cmd = [
			sys.executable,
			"-m",
			"emg2pose.test_analysis",
			f"data_location={home_dataset_dir}",
			f"experiment={experiment}",
			f"checkpoint={home_checkpoints_dir / checkpoint_name}",
			"num_workers=12",
		]
		print(f"$ {' '.join(cmd)}")
		subprocess.run(cmd, cwd=REMOTE_REPO_PATH, env=env, check=True)


@app.local_entrypoint()
def main(
	download_dataset: bool = True,
	run_test_analysis: bool = True,
	experiment: str = "tracking_vemg2pose",
	checkpoint_name: str = "tracking_vemg2pose.ckpt",
):
	# Temporarily using .remote() to see startup errors
	# Switch back to .spawn() once it's working
	setup_and_run_test.remote(
		download_dataset=download_dataset,
		run_test_analysis=run_test_analysis,
		experiment=experiment,
		checkpoint_name=checkpoint_name,
	)
