import traceback
from pathlib import Path
import torch
from detection_core import (
    set_seed, parse_coco_json_thistle, split_dataset_indices,
    COCODetectionDataset, run_detection_training, IN_KAGGLE
)


class ThistleConfig:
    SEED = 42
    if IN_KAGGLE:
        DATA_ROOT = Path("")
        JSON_FILE = ""
        IMAGE_FOLDER = ""
        BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        NUM_WORKERS = 2
        CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
        FINAL_MODEL_PATH = Path("/kaggle/working/final_weed_detection_model.pth")
    else:
        DATA_ROOT = Path("")
        JSON_FILE = ""
        IMAGE_FOLDER = ""
        BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        NUM_WORKERS = 1
        CHECKPOINT_DIR = Path("")
        FINAL_MODEL_PATH = Path("")
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    TRAIN_SPLIT_RATIO = 0.8
    VAL_SPLIT_RATIO = 0.2
    SUBSET_PERCENTAGE = 75
    BACKBONE_TYPE = 'pretrained'
    BACKBONE_ARCH = 'resnet50'
    WANDB_ARTIFACT = ""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True
    GRADIENT_CLIPPING = 1.0
    NUM_CLASSES = 2
    MIN_SIZE = 800
    MAX_SIZE = 1333
    ANCHOR_SIZES = ((35, 64, 93, 141, 199, 300),)
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
    CLASS_NAMES = ["background", "weed: cirsium vulgare"]
    TEST_IMAGES = [
        "20241111_144806_jpg.rf.0a764c4b6f1d0a7f784bb9ac68957c8b.jpg",
        "20241111_143011_jpg.rf.51c23edd496c1abbe9b04e04051fe0b3.jpg",
        "20241111_142813_jpg.rf.1bde175a5030f7f4f76aa35b0d449a37.jpg",
        "20241112_131609_jpg.rf.c9dc349257c6d60ac1584bc1dc692b78.jpg",
        "20241111_143526_jpg.rf.72280823e74a5609a0a33219fad9e6fa.jpg",
        "20241111_143445_jpg.rf.761005de57b9f534fc86ff5845a90189.jpg",
        "20241111_143414_jpg.rf.210efa5de04861d3661add68628ef531.jpg",
        "20241111_143057_jpg.rf.ecc1469a21de37070519afd8a6eb9e42.jpg",
        "20241112_131603_jpg.rf.71c719947dfe33b6ba2ca1598b6629f6.jpg",
        "20241111_142957_jpg.rf.1a2c6695a83840cac61ae356c19eb6a2.jpg",
        "20241112_131444_jpg.rf.5a8b36b95aa1fbc22f9e169f242fbe72.jpg",
        "20241112_131835_jpg.rf.f4eb233252f19d7740ad4bc420f50d20.jpg",
        "20241112_131746_jpg.rf.6278ba79f790f2e4ddb387f7145d435.jpg",
        "20241111_144438_jpg.rf.e651e6050b8133a9105fe959a798f999.jpg",
        "20241111_142923_jpg.rf.28848b1c9ac80c74151aae260ca686e8.jpg",
        "20241111_144653_jpg.rf.4900ac3a9d370c300dc59874a20f52b0.jpg",
        "20241111_144052_jpg.rf.bd6a12194faf7eca40540589cb7a74b0.jpg",
        "20241111_142818_jpg.rf.4d50b490f34f32407e2ec495ee13d2bd.jpg",
        "20241111_142816_jpg.rf.70f85ba734209e29c0b72ee0ef815f19.jpg",
        "20241111_142927_jpg.rf.4714a5b95cd9d2d9dfc73bf7fbad387d.jpg",
        "20241111_143357_jpg.rf.5d49c844237e4ab0cba37f66b591ec49.jpg",
        "20241111_143426_jpg.rf.a3645f514f0d1d23778a091d4c62b26d.jpg",
        "20241111_144317_jpg.rf.4dc16821a41c115bfef4d3715c355f29.jpg",
        "20241111_144712_jpg.rf.b3e985b22b9bd27b44a8a378d6542ba6.jpg",
    ]

    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")


def main():
    config = ThistleConfig()
    set_seed(config.SEED)
    try:
        json_path = config.DATA_ROOT / config.JSON_FILE
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        all_annotations, img_id_to_name = parse_coco_json_thistle(
            json_path, test_images=config.TEST_IMAGES
        )
        all_image_ids = list(all_annotations.keys())
        if len(all_image_ids) == 0:
            raise ValueError(f"No non-test images found in: {json_path}")
        train_ids, val_ids = split_dataset_indices(
            all_image_ids, train_ratio=config.TRAIN_SPLIT_RATIO, seed=config.SEED
        )
        train_dataset = COCODetectionDataset(
            config.DATA_ROOT, config, all_annotations, img_id_to_name,
            split_ids=train_ids, seed_offset=0,
            image_folder=config.IMAGE_FOLDER if config.IMAGE_FOLDER else None
        )
        val_dataset = COCODetectionDataset(
            config.DATA_ROOT, config, all_annotations, img_id_to_name,
            split_ids=val_ids, seed_offset=1000,
            image_folder=config.IMAGE_FOLDER if config.IMAGE_FOLDER else None
        )
        run_detection_training(config, train_dataset, val_dataset)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
