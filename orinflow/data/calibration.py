"""Calibration data preparation utilities."""

import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import yaml

from orinflow.config import DATA_DIR, CALIB_DIR, DEFAULT_CALIB_SAMPLES, DEFAULT_INPUT_SIZE

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
        _download_and_extract(download_url, dataset_dir_name)

    if not split_dir:
        raise ValueError(f"Config file {yaml_path} missing '{split}' field")

    # Resolve image directory
    if os.path.isabs(split_dir):
        images_dir = Path(split_dir)
    else:
        images_dir = dataset_root / split_dir

    # Find images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(images_dir.glob(ext)))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

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


def prepare_calibration_data(
    yaml_file: str | Path,
    split: str = "train",
    num_images: int = DEFAULT_CALIB_SAMPLES,
    input_size: int = DEFAULT_INPUT_SIZE,
    output_name: str | None = None,
) -> Path:
    """
    Prepare calibration data for quantization.

    Args:
        yaml_file: Path to dataset YAML file
        split: Dataset split to use
        num_images: Number of calibration images
        input_size: Model input size
        output_name: Output file name (without extension), defaults to {yaml_stem}_calib

    Returns:
        Path to saved calibration data (.npy)
    """
    yaml_path = Path(yaml_file)
    if output_name is None:
        output_name = yaml_path.stem
    output_path = CALIB_DIR / f"{output_name}.npy"

    # Load image paths
    image_paths = load_dataset_images(yaml_path, split, num_images)

    # Process images
    print(f"Preprocessing {len(image_paths)} images...")
    calib_data = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: Cannot read image: {p}")
            continue
        img_tensor = preprocess_for_yolo(img, input_size=(input_size, input_size))
        calib_data.append(img_tensor)

    if len(calib_data) == 0:
        raise ValueError("No images successfully processed")

    calib_blob = np.concatenate(calib_data, axis=0)
    print(f"Calibration data shape: {calib_blob.shape}")
    print(f"Data type: {calib_blob.dtype}")
    print(f"Value range: [{calib_blob.min():.3f}, {calib_blob.max():.3f}]")

    # Save
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(output_path, calib_blob)
    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path
