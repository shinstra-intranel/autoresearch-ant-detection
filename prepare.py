"""
Fixed evaluation harness, data loading, and budget enforcement.
DO NOT MODIFY — this file is read-only (enforced via VS Code settings).

<!-- HUMAN: Implement the four sections below for your specific problem. -->
"""

import os
import time
import torch

import PIL.Image
import PIL.ImageDraw
import numpy as np

import sys
from pathlib import Path

from transformers import TrainerCallback
from transformers import AutoImageProcessor

import albumentations as A

from typing import List, Dict, Any

from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format

import logging

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify once set)
# ---------------------------------------------------------------------------

# HUMAN: Set these for your problem domain.

TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
                          # Alternatives: set to None if using step-based budget
MAX_STEPS = None          # max optimizer steps (set this OR TIME_BUDGET, not both)
EVAL_SAMPLES = 10000      # number of samples for validation evaluation

CLASS_NAMES = ["Ant"]
CLASS_MAPPING = {"Ant": 0}

# Create label mappings
ID2LABEL = {i: label for i, label in enumerate(CLASS_NAMES)}
LABEL2ID = {label: i for i, label in enumerate(CLASS_NAMES)}

# HUMAN: Add any other constants your solution needs to import.
# Examples:
#   MAX_SEQ_LEN = 2048      # for language models
#   IMAGE_SIZE = 224         # for vision models
#   NUM_CLASSES = 10         # for classification
#   ACTION_SPACE = 4         # for RL


# ---------------------------------------------------------------------------
# Utilities (fixed, do not modify once set)
# ---------------------------------------------------------------------------

class TimeBudgetCallback(TrainerCallback):
    def __init__(self, time_budget=TIME_BUDGET):
        self.time_budget = time_budget
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_budget:
            control.should_training_stop = True
        return control

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self  # Returns itself, allowing access in `with` block

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start  # Store execution time

    def __float__(self):
        return self.elapsed

    def __call__(self):
        return self.elapsed

    def __repr__(self):
        # Provides a readable output when printed
        return f"{self.elapsed:.3f}"

    def __str__(self):
        return f"{self.elapsed:.3f}"

def generate_run_id():
    """Generates UTC timestamp-based run ID for logging and outputs."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


_logging_initialised = False


def setup_logging(log_file: str = "run.log") -> logging.Logger:
    """Configure root logger to write to log_file (fresh each run) and stdout.

    Safe to call multiple times — only the first call deletes and recreates the
    log file; subsequent calls are no-ops so repeated invocations (e.g. from
    HuggingFace internals) cannot truncate the file mid-run.
    """
    global _logging_initialised
    if _logging_initialised:
        return logging.getLogger(__name__)

    _logging_initialised = True

    # Delete the old log file so the run starts fresh, then open in append mode
    # so any further FileHandler creations won't truncate it.
    Path(log_file).unlink(missing_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Suppress verbose logging from libraries we use (e.g. HuggingFace, httpx)
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class AntDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, processor: AutoImageProcessor, transform=None):
        """
        Load the dataset split into memory.

        Args:
            split (str): which split to load ("train" or "test")

        Returns:
            AntDataset: a PyTorch Dataset object for the specified split
        """
        self.split = split
        self.processor = processor
        self.transform = transform

        self.image_folder = f".data/{split}/images"
        self.labels_folder = f".data/{split}/bboxes" 

        # List valid image-label pairs
        self.filenames = []

        image_filenames = sorted(os.listdir(self.image_folder))  # e.g. image1.jpg
        label_filenames = sorted(os.listdir(self.labels_folder))   # e.g. bbox1.txt

        for image_filename in image_filenames:
            label_filename = os.path.splitext(image_filename)[0].replace("image", "bbox") + ".txt"
            if label_filename in label_filenames:
                self.filenames.append((image_filename, label_filename))
        
    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def format_image_annotations_as_coco(image_id, categories, boxes):
        """
        Format one set of image annotations to the COCO format for image processor.
        
        Args:
            image_id: image id
            categories: list of categories/class labels
            boxes: list of bounding boxes in COCO format ([x, y, width, height])
            
        Returns:
            dict: formatted annotations for HuggingFace image processor
        """
        annotations = []
        for category, bbox in zip(categories, boxes):
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": list(bbox),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            annotations.append(formatted_annotation)
        
        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def load_image(self, filename):
        image = PIL.Image.open(os.path.join(self.image_folder, filename)).convert("RGB")
        return np.array(image)
    
    def load_labels(self, filename):
        bboxes = []
        with open(os.path.join(self.labels_folder, filename), "r") as f:
            for line in f:
                x_min, y_min, x_max, y_max = map(float, line.strip().split())
                x, y, width, height = x_min, y_min, x_max - x_min, y_max - y_min
                bboxes.append((x, y, width, height))
        class_labels = [0] * len(bboxes)  # Assuming all bounding boxes are ants (class 0)
        return bboxes, class_labels
    
    def __getitem__(self, idx):
        
        image_filename, label_filename = self.filenames[idx]

        image = self.load_image(image_filename)
        bboxes, class_labels = self.load_labels(label_filename)

        # Apply transformations (augmentations)
        if self.transform:
            transformed = self.transform(
                image=image, 
                bboxes=bboxes, 
                class_labels=class_labels
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        
        # Format annotations in COCO format for image_processor
        formatted_annotations = self.format_image_annotations_as_coco(
            idx, class_labels, bboxes
        )

        # Apply the image processor transformations: resizing, rescaling, normalization
        inputs = self.processor(
            images=image, 
            annotations=formatted_annotations, 
            return_tensors="pt"
        )

        # Image processor expands batch dimension, squeeze it
        inputs = {k: v[0] for k, v in inputs.items()}

        return inputs
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        return data
    

def get_train_dataset(processor: AutoImageProcessor, transform: A.Compose) -> AntDetectionDataset:
    """Setup training dataset with augmentations"""
    dataset = AntDetectionDataset(
        split="train",
        processor=processor,
        transform=transform
    )
    return dataset

def get_test_dataset(processor: AutoImageProcessor) -> AntDetectionDataset:
    """Setup validation dataset without augmentations"""

    # Identity transform (no augmentations) with proper bbox_params for image processor
    transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco", 
            label_fields=["class_labels"], 
            clip=True, 
            min_area=1, 
            min_width=1, 
            min_height=1
        ),
    )

    dataset = AntDetectionDataset(
        split="test",
        processor=processor,
        transform=transform,
    )

    return dataset

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE once implemented — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, processor=None, valid_dataset=None, **kwargs):
    """
    Compute the evaluation metric on the validation set.

    Args:
        model: the trained model
        processor: the image processor (if not provided, will look in kwargs)
        valid_dataset: the validation dataset (if not provided, will look in kwargs)

    Returns:
        float: the metric value (lower is better if minimizing,
               higher is better if maximizing — set direction in identity.md)

    """
    if processor is None:
        processor = kwargs.get("processor")
    if valid_dataset is None:
        valid_dataset = kwargs.get("valid_dataset") or kwargs.get("test_dataset")

    if processor is None or valid_dataset is None:
        raise ValueError("evaluate requires both processor and valid_dataset")

    device = next(model.parameters()).device
    num_samples = min(len(valid_dataset), EVAL_SAMPLES)
    subset = torch.utils.data.Subset(valid_dataset, range(num_samples))
    dataloader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        collate_fn=getattr(valid_dataset, "collate_fn", AntDetectionDataset.collate_fn),
        num_workers=0,
    )

    evaluator = MeanAveragePrecision(box_format="xyxy")
    evaluator.warn_on_many_detections = False

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)

        target_sizes = []
        targets = []
        for label in labels:
            size = label.get("orig_size")
            if size is None:
                size = label.get("size")
            if size is None:
                raise ValueError("Expected labels to include 'orig_size' or 'size'.")

            size = size.detach().cpu()
            height, width = int(size[0]), int(size[1])
            target_sizes.append([height, width])

            scale = torch.tensor([width, height, width, height], dtype=label["boxes"].dtype)
            targets.append(
                {
                    "boxes": center_to_corners_format(label["boxes"].detach().cpu()) * scale,
                    "labels": label["class_labels"].detach().cpu(),
                }
            )

        predictions = processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=torch.tensor(target_sizes),
        )
        predictions = [{k: v.detach().cpu() for k, v in prediction.items()} for prediction in predictions]

        evaluator.update(predictions, targets)

    metrics = evaluator.compute()
    return float(metrics["map_50"].item())

# ---------------------------------------------------------------------------
# Main: one-time data preparation (run once before experiments)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Generate test batch with annotations drawn for manual verification of data pipeline.

    def _pixel_values_to_pil_image(pixel_values: torch.Tensor, processor: AutoImageProcessor) -> PIL.Image.Image:
        image = pixel_values.detach().cpu().float()

        image_mean = getattr(processor, "image_mean", None)
        image_std = getattr(processor, "image_std", None)
        do_normalize = getattr(processor, "do_normalize", image_mean is not None and image_std is not None)
        if do_normalize and image_mean is not None and image_std is not None:
            mean = torch.tensor(image_mean, dtype=image.dtype).view(-1, 1, 1)
            std = torch.tensor(image_std, dtype=image.dtype).view(-1, 1, 1)
            image = image * std + mean

        do_rescale = getattr(processor, "do_rescale", False)
        rescale_factor = getattr(processor, "rescale_factor", None)
        if do_rescale and rescale_factor:
            image = image / rescale_factor
        elif image.max() <= 1.0:
            image = image * 255.0

        image = image.clamp(0, 255)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image_array = image.permute(1, 2, 0).round().byte().numpy()
        return PIL.Image.fromarray(image_array)


    def _labels_to_xyxy_boxes(labels: Dict[str, Any]) -> torch.Tensor:
        boxes = labels.get("boxes")
        if boxes is None:
            return torch.empty((0, 4), dtype=torch.float32)

        boxes = boxes.detach().cpu()
        if boxes.numel() == 0:
            return torch.empty((0, 4), dtype=torch.float32)

        size = labels.get("size")
        if size is None:
            size = labels.get("orig_size")
        if size is None:
            raise ValueError("Expected labels to include 'size' or 'orig_size'.")

        if isinstance(size, torch.Tensor):
            size = size.detach().cpu().tolist()

        height, width = int(size[0]), int(size[1])
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        return center_to_corners_format(boxes) * scale


    def export_test_batch(dataset: AntDetectionDataset, batch_size: int = 4) -> List[str]:
        if len(dataset) == 0:
            raise ValueError("Cannot export a batch from an empty dataset.")

        output_dir = os.path.join(".data", "test_batches", dataset.split)
        os.makedirs(output_dir, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        batch = next(iter(loader))

        saved_paths = []
        filenames = getattr(dataset, "filenames", [])

        for index, (pixel_values, labels) in enumerate(zip(batch["pixel_values"], batch["labels"])):
            image = _pixel_values_to_pil_image(pixel_values, dataset.processor)
            draw = PIL.ImageDraw.Draw(image)

            for x0, y0, x1, y1 in _labels_to_xyxy_boxes(labels).tolist():
                draw.rectangle((x0, y0, x1, y1), outline="red", width=3)

            if index < len(filenames):
                image_filename = filenames[index][0]
                output_name = f"{os.path.splitext(image_filename)[0]}.png"
            else:
                output_name = f"sample_{index:03d}.png"

            output_path = os.path.join(output_dir, output_name)
            image.save(output_path)
            saved_paths.append(output_path)

        return saved_paths

    processor = AutoImageProcessor.from_pretrained(
        "PekingU/rtdetr_v2_r18vd" ,
        do_resize=True,
        size={"height": 640, "width": 640},
        use_fast=True,
    )

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5)],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], clip=True, min_area=25, min_width=1, min_height=1),
    )

    train_dataset = get_train_dataset(processor=processor, transform=transform)

    export_test_batch(train_dataset, batch_size=4)
