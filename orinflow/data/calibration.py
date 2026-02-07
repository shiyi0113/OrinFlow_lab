"""Calibration data preparation utilities."""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import yaml

from orinflow.config import DATA_DIR, DEFAULT_CALIB_SAMPLES, DEFAULT_INPUT_SIZE

from .preprocessing import preprocess_for_yolo


def load_dataset_images(
    yaml_file: str | Path,
    split: str = "train",
    num_images: int = DEFAULT_CALIB_SAMPLES,
) -> list[Path]:
    """
    Load image paths from dataset YAML configuration.

    Args:
        yaml_file: Path to dataset YAML file
        split: Dataset split to use (train, val, test)
        num_images: Number of images to load

    Returns:
        List of image paths
    """
    yaml_path = Path(yaml_file)
    if not yaml_path.is_absolute():
        yaml_path = DATA_DIR / yaml_path

    print(f"Loading config: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    dataset_dir_name = cfg.get("path")
    split_dir = cfg.get(split)
    download_url = cfg.get("download")

    if not dataset_dir_name:
        raise ValueError(f"Config file {yaml_path} missing 'path' field")

    dataset_root = DATA_DIR / dataset_dir_name

    # Download dataset if not exists
    if not dataset_root.exists():
        if not download_url:
            raise ValueError(f"Dataset {dataset_root} not found and no download URL provided")
        download_str = download_url.strip()
        if download_str.startswith("http"):
            _download_and_extract(download_str, dataset_dir_name)
        elif download_str.endswith(".sh"):
            _run_shell_script(download_str, dataset_dir_name)
        else:
            _exec_download_script(download_url, cfg)

    if not split_dir:
        raise ValueError(f"Config file {yaml_path} missing '{split}' field")

    # Resolve image directory
    split_dirs = split_dir if isinstance(split_dir, list) else [split_dir]
    
    image_files = []
    for sd in split_dirs:
        if os.path.isabs(sd):
            split_path = Path(sd)
        else:
            split_path = dataset_root / sd

        if split_path.is_file() and split_path.suffix == ".txt":
            # txt mode
            with open(split_path) as f:
                paths = [Path(line.strip()) for line in f if line.strip()]
            paths = [p if p.is_absolute() else dataset_root / p for p in paths]
            image_files.extend([p for p in paths if p.exists()])
        else:
            # dir mode
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(list(split_path.glob(ext)))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {split_path}")

    # Random sample
    np.random.seed(42)
    if len(image_files) > num_images:
        image_files = np.random.choice(image_files, num_images, replace=False).tolist()

    print(f"Found {len(image_files)} images for calibration")
    return image_files


def _download_and_extract(url: str, dataset_name: str) -> None:
    """Download and extract dataset."""
    zip_path = DATA_DIR / f"{dataset_name}.zip"
    dataset_dir = DATA_DIR / dataset_name

    print(f"Downloading from {url}...")
    try:
        urlretrieve(
            url,
            zip_path,
            reporthook=lambda count, block_size, total_size: print(
                f"\rDownloading: {int(count * block_size * 100 / total_size)}%", end=""
            ),
        )
        print("\nDownload complete.")
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Download failed: {e}")

    # Extract
    print(f"Extracting to {DATA_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(dataset_dir)

    # Handle nested directory
    extracted_items = [item for item in dataset_dir.iterdir() if not item.name.startswith((".", "_"))]
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        nested_folder = extracted_items[0]
        for sub_item in nested_folder.iterdir():
            shutil.move(str(sub_item), str(dataset_dir))
        nested_folder.rmdir()

    if zip_path.exists():
        zip_path.unlink()
    print("Extraction complete.")


def _run_shell_script(script_path: str, dataset_name: str) -> None:
    """Run a shell script to download dataset."""
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running shell script: {script_path}")
    subprocess.run(["bash", script_path], cwd=str(dataset_dir), check=True)
    print("Shell script complete.")


def _exec_download_script(script: str, yaml_cfg: dict) -> None:
    """Execute a download script from YAML config."""
    print("Running download script...")
    cfg = yaml_cfg.copy()
    cfg["path"] = str(DATA_DIR / cfg["path"])
    exec(script, {"yaml": cfg})
    print("Download script complete.")


def get_calibration_data(
    yaml_file: str | Path,
    split: str = "train",
    num_images: int = DEFAULT_CALIB_SAMPLES,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    yaml_path = Path(yaml_file)

    # Load image paths
    image_paths = load_dataset_images(yaml_path, split, num_images)

    # Process images
    print(f"Preprocessing {len(image_paths)} images...")
    calib_data = []
    
    for i, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: Cannot read image: {p}")
            continue
        
        # Preprocess: Letterbox -> CHW -> Normalize
        img_tensor = preprocess_for_yolo(img, input_size=(input_size, input_size))
        calib_data.append(img_tensor)
        
        # Optional: Print progress for large datasets
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} images...")

    if len(calib_data) == 0:
        raise ValueError("No images successfully processed")

    calib_blob = np.concatenate(calib_data, axis=0)
    
    print(f"Calibration data ready.")
    print(f"  Shape: {calib_blob.shape}")
    print(f"  Type:  {calib_blob.dtype}")
    print(f"  Range: [{calib_blob.min():.3f}, {calib_blob.max():.3f}]")

    return calib_blob