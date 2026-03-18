"""
Training script — the agent modifies this. Everything is fair game.
"""
import torch
import logging
import albumentations as A

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import TrainingArguments, Trainer, PrinterCallback

from prepare import (
    TIME_BUDGET,
    ID2LABEL,
    LABEL2ID,
    TimeBudgetCallback,
    AntDetectionDataset,
    Timer,
    evaluate,
    get_train_dataset,
    get_test_dataset,
    generate_run_id,
    setup_logging,
)

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)  # KEEP: suppress benign warnings for cleaner logs

# ---------------------------------------------------------------------------
# Setup logging - (do not modify this section)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)    # KEEP

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits these)
# ---------------------------------------------------------------------------

DEVICE = 0 # GPU
WORKERS = 2

LEARNING_RATE = 1e-3
BATCH_SIZE = 6
EPOCHS = 100
WARMUP_EPOCHS = 10
MAX_GRAD_NORM = 0.1

MODEL_NAME = "PekingU/rtdetr_v2_r18vd"  # KEEP: 
FREEZE_BACKBONE = True

INPUT_SIZE = 640  # Specific to model


# ---------------------------------------------------------------------------
# Model (agent edits this)
# ---------------------------------------------------------------------------

def get_image_processor(model_name: str) -> AutoImageProcessor:
    """Setup image processor for the model"""
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        do_resize=True,
        size={"height": INPUT_SIZE, "width": INPUT_SIZE},
        use_fast=True,
    )
    return processor

def get_model(model_name: str) -> AutoModelForObjectDetection:
    """Setup RT-DETR model for object detection"""
    
    # Load model
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    
    # Optionally freeze backbone
    if FREEZE_BACKBONE:
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    
    return model

# ---------------------------------------------------------------------------
# Training dataset augmentation (agent edits this)
# ---------------------------------------------------------------------------

train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=3,      # ~0.015 * 180
            sat_shift_limit=30,     # ~0.7 * 255 (scaled down for stability)
            val_shift_limit=30,     # ~0.4 * 255 (scaled down for stability)
            p=0.7
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
    ],
    bbox_params=A.BboxParams(
        format="coco", 
        label_fields=["class_labels"], 
        clip=True, 
        min_area=25, 
        min_width=1, 
        min_height=1
    ),
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():

    run_id = generate_run_id()

    setup_logging() # KEEP: must be inside train() so worker processes don't call it

    with Timer() as total_timer:    

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Model
            processor = get_image_processor(MODEL_NAME)

            model = get_model(MODEL_NAME).to(device)
            model.train()

            # Data
            train_dataset = get_train_dataset(processor=processor, transform=train_transforms)
            test_dataset = get_test_dataset(processor=processor)

            # Training
            training_args = TrainingArguments(
                num_train_epochs=EPOCHS,
                max_grad_norm=MAX_GRAD_NORM,
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=BATCH_SIZE,
                dataloader_num_workers=WORKERS,
                dataloader_drop_last=True,
                logging_strategy="steps",               # KEEP: log training loss/lr every step
                logging_steps=1,                        # KEEP: 
                report_to=["tensorboard"],              # KEEP: Keep training logs for Human review
                save_strategy="no",                     # KEEP: Disable checkpoint saving
                output_dir=f".trainer_output/{run_id}", # KEEP: Where automated logs and outputs go
                disable_tqdm=True,                      # KEEP: Reduce overhead and clutter in logs
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=AntDetectionDataset.collate_fn,
                train_dataset=train_dataset,
                processing_class=processor,
                callbacks=[
                    TimeBudgetCallback(TIME_BUDGET),
                ],
            )

            trainer.remove_callback(PrinterCallback)    # KEEP: supress per-step logging to stdout. Data is still logged to TensorBoard.

            with Timer() as training_timer:
                trainer.train()

            # Evaluation
            model.eval()
            metric = evaluate(model, processor, test_dataset)  # KEEP: use the same evaluation function as in the baseline for consistency

            # VRAM usage
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0  # KEEP: 

        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            return

    # KEEP: DO NOT MODIFY PAST THIS POINT.
    # KEEP: The number of lines in the "Reading Results" command has been configured to grab exactly these lines, so changing the output format will break result collection.

    logger.info(f"run_id:           {run_id}")                          # KEEP:
    logger.info(f"model_name:       {MODEL_NAME}")                      # KEEP:
    logger.info(f"device:           {device}")                          # KEEP:
    logger.info(f"val_metric:       {metric:.6f}")                      # KEEP:
    logger.info(f"training_seconds: {training_timer.elapsed:.1f}")      # KEEP:
    logger.info(f"total_seconds:    {total_timer.elapsed:.1f}")         # KEEP:
    logger.info(f"peak_vram_mb:     {peak_vram_mb:.1f}")                # KEEP:
    logger.info(f"num_steps:        {trainer.state.global_step}")       # KEEP:
    logger.info(f"num_epochs:       {trainer.state.epoch:.2f}")         # KEEP:


if __name__ == "__main__":
    train()
