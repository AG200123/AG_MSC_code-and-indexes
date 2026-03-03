import traceback
from pathlib import Path
import torch
from detection_core import (
    set_seed, parse_coco_json_sorghum, COCODetectionDataset,
    run_detection_training, IN_KAGGLE
)


class SorghumConfig:
    SEED = 42
    if IN_KAGGLE:
        DATA_ROOT = Path("")
        BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        NUM_WORKERS = 4
        CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
        FINAL_MODEL_PATH = Path("/kaggle/working/final_sorghum_weed_detection_model.pth")
    else:
        DATA_ROOT = Path("")
        BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        NUM_WORKERS = 2
        CHECKPOINT_DIR = Path("")
        FINAL_MODEL_PATH = Path("")
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    SUBSET_PERCENTAGE = 75
    BACKBONE_TYPE = 'pretrained'
    BACKBONE_ARCH = 'resnet50'
    WANDB_ARTIFACT = ""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True
    GRADIENT_CLIPPING = 1.0
    NUM_CLASSES = 4
    MIN_SIZE = 800
    MAX_SIZE = 1333
    ANCHOR_SIZES = ((32, 40, 48, 112, 160),)
    ASPECT_RATIOS = ((1.0, 0.5, 2.0),)
    WANDB_PROJECT = ""
    WANDB_ENTITY = None
    WANDB_RUN_NAME = ""
    WANDB_LOG_IMAGES = False
    WANDB_LOG_GRADIENTS = False
    WANDB_LOG_MODEL = True
    CHECKPOINT_SAVE_FREQUENCY = 10
    SAVE_FINAL_MODEL = False
    FREEZE_BACKBONE = True
    CLASS_NAMES = ["background", "Sorghum", "Monocot weeds", "Dicot weeds"]

    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")


def main():
    config = SorghumConfig()
    set_seed(config.SEED)
    try:
        train_dir = config.DATA_ROOT / "Train"
        valid_dir = config.DATA_ROOT / "Validate"
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        if not valid_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {valid_dir}")
        train_json = train_dir / "TrainSorghumWeed_coco.json"
        val_json = valid_dir / "ValidateSorghumWeed_coco.json"
        train_annotations, train_id_to_name = parse_coco_json_sorghum(train_json)
        val_annotations, val_id_to_name = parse_coco_json_sorghum(val_json)
        train_dataset = COCODetectionDataset(
            train_dir, config, train_annotations, train_id_to_name, seed_offset=0
        )
        val_dataset = COCODetectionDataset(
            valid_dir, config, val_annotations, val_id_to_name, seed_offset=1000
        )
        run_detection_training(config, train_dataset, val_dataset)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
