import traceback
from pathlib import Path
import torch
from detection_core import (
    set_seed, parse_coco_json_vcd, split_dataset_indices,
    COCODetectionDataset, run_detection_training, IN_KAGGLE
)


class VCDConfig:
    SEED = 42
    if IN_KAGGLE:
        DATA_ROOT = Path("")
        JSON_FILE = ""
        IMAGE_FOLDER = ""
        BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        NUM_WORKERS = 2
        CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
        FINAL_MODEL_PATH = Path("/kaggle/working/final_vcd_detection_model.pth")
        LOCAL_BACKBONE_PATH = ""
    else:
        DATA_ROOT = Path("")
        JSON_FILE = ""
        IMAGE_FOLDER = ""
        BATCH_SIZE = 4
        VAL_BATCH_SIZE = 4
        NUM_WORKERS = 1
        CHECKPOINT_DIR = Path("")
        FINAL_MODEL_PATH = Path("")
        LOCAL_BACKBONE_PATH = ""
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
    ANCHOR_SIZES = ((32, 40, 48, 112, 160),)
    ASPECT_RATIOS = ((1.0, 0.5, 2.0),)
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5
    WANDB_PROJECT = ""
    WANDB_ENTITY = None
    WANDB_RUN_NAME = ""
    WANDB_LOG_IMAGES = False
    WANDB_LOG_GRADIENTS = False
    WANDB_LOG_MODEL = True
    CHECKPOINT_SAVE_FREQUENCY = 10
    SAVE_FINAL_MODEL = False
    FREEZE_BACKBONE = True
    CLASS_NAMES = ["background", "crop: allium ampeloprasum"]
    TEST_IMAGES = [
        "2019-07-03_liposthey_leek_2_000114_jpg.rf.24a61337967702ed5b3d8864b249593b.jpg",
        "2019-07-03_liposthey_leek_2_000025_jpg.rf.9399f5b9d1fe746163e7bc303e990b52.jpg",
        "2021-03-29_liposthey_leek_1_000054_jpg.rf.79122333926f6943817558f9e0db0018.jpg",
        "2021-03-29_liposthey_leek_1_000023_jpg.rf.7c45cf849093acc0830c05f1c5f67c30.jpg",
        "2021-03-29_liposthey_leek_1_000001_jpg.rf.665916b7e8d80da074b1cfd81447dce9.jpg",
        "2019-07-03_liposthey_leek_3_000027_jpg.rf.fdba81fbc0649e567321e65810b9f249.jpg",
        "2019-07-03_liposthey_leek_2_000104_jpg.rf.f45c5fabf20d4ab98c15ba1a92971111.jpg",
        "2021-07-20_lanxade_leek_3_000038_jpg.rf.4b3aa66f5c4c1973c9ca7c9ea7d1f5e5.jpg",
        "2019-07-03_liposthey_leek_2_000089_jpg.rf.a8468aa6adacad2b6e18ef40028adef3.jpg",
        "2021-03-29_liposthey_leek_1_000205_jpg.rf.f495609f999eb4ec204caddb25f539fd.jpg",
        "2019-07-03_liposthey_leek_2_000032_jpg.rf.8298270933ef435a86073c2e10fdfcd8.jpg",
        "2019-07-03_liposthey_leek_2_000030_jpg.rf.8177f6a2a6ae842910dbd5639392c9be.jpg",
        "2019-07-03_liposthey_leek_2_000095_jpg.rf.6ebcf01e7746cdc7bb86d95a1731c603.jpg",
        "2019-07-03_liposthey_leek_4_000026_jpg.rf.f7c91eaf931da32f151b7a70f52b3d5b.jpg",
        "2021-03-29_liposthey_leek_1_000011_jpg.rf.286a3f8e63c14d5b8eb8792c6034ea73.jpg",
        "2021-07-20_lanxade_leek_2_000014_jpg.rf.185c8b7dc960816cd7654d2bb370744d.jpg",
        "2019-07-03_liposthey_leek_2_000027_jpg.rf.48eae82a590a61db3ec5f59429f53dbd.jpg",
        "2021-07-20_lanxade_leek_4_000009_jpg.rf.120afca37650ca5c4a40eb683c71b2c5.jpg",
        "2019-07-03_liposthey_leek_4_000006_jpg.rf.8ed8d30517420953e594ad2edf6a552a.jpg",
        "2021-03-29_liposthey_leek_1_000202_jpg.rf.b7f6655f240ee9742a13a6a0e1cae314.jpg",
        "2019-07-03_liposthey_leek_4_000028_jpg.rf.46ad0c1062e771840aa1dfa77a7b8fd1.jpg",
        "2021-05-24_bordeaux_leek_2_000007_jpg.rf.c3bc368ae406f8fc2249c6f68c88cc8c.jpg",
        "2021-03-29_liposthey_leek_1_000057_jpg.rf.51dd1936197d9159e8abcf0a07d09857.jpg",
        "2019-07-03_liposthey_leek_2_000006_jpg.rf.510da6aab96ec39845d07f390c0c98c3.jpg",
        "2019-07-03_liposthey_leek_3_000048_jpg.rf.9b129defc615b09f668a7dd5b735991f.jpg",
        "2021-03-29_liposthey_leek_1_000121_jpg.rf.6491760a8c732254aff4f6720c45b5d7.jpg",
        "2019-07-03_liposthey_leek_3_000044_jpg.rf.4f62dc2b65c10d1a068bb5c854fbea8f.jpg",
        "2019-07-03_liposthey_leek_4_000023_jpg.rf.ba101113f75c19df0ae1cdaeb294c483.jpg",
        "2021-03-29_liposthey_leek_1_000117_jpg.rf.e4aa89d2491e0ec88cf616a4447e37bd.jpg",
        "2019-07-03_liposthey_leek_2_000094_jpg.rf.bb8c88ebbe88c7f1360f9de52ed373e0.jpg",
        "2021-03-29_liposthey_leek_1_000162_jpg.rf.4302e9167ea71ddae6f941b4a12c4724.jpg",
        "2019-07-03_liposthey_leek_2_000099_jpg.rf.dd47f663ba1883543e9297b7bf900d47.jpg",
        "2021-03-29_liposthey_leek_1_000140_jpg.rf.7c8d7d0d0e2f6546112b600884298833.jpg",
        "2021-03-29_liposthey_leek_1_000125_jpg.rf.852b12372ba79bc26ec9e6afca4a379c.jpg",
        "2021-03-29_liposthey_leek_1_000043_jpg.rf.85d367db90a5f5f6802239c4f34a47e6.jpg",
        "2019-07-03_liposthey_leek_2_000044_jpg.rf.c3e9e4ce66bd7bbf4282836623c08371.jpg",
        "2021-05-24_bordeaux_leek_3_000007_jpg.rf.837b32f857fdc30309b4ddf50cac3767.jpg",
        "2021-07-20_lanxade_leek_3_000029_jpg.rf.5ac028c409e8215a5e0066f14464b3b3.jpg",
        "2019-07-03_liposthey_leek_3_000012_jpg.rf.323d4b53a666d91413f946a3b47580f6.jpg",
        "2021-03-29_liposthey_leek_1_000160_jpg.rf.3348508fe1f09d1031c0cfabed5b39cd.jpg",
        "2019-07-03_liposthey_leek_2_000080_jpg.rf.0412721c4330f4d4caec56b7e5950d84.jpg",
        "2021-07-20_lanxade_leek_4_000000_jpg.rf.5a0738abaf5c57add8276015ebe12613.jpg",
        "2021-03-29_liposthey_leek_1_000073_jpg.rf.7126bf318dd64991b54c3e574e2a3255.jpg",
        "2021-03-29_liposthey_leek_1_000143_jpg.rf.864d0999cafc6766a0ac4bf6c9cafe37.jpg",
        "2021-07-20_lanxade_leek_4_000026_jpg.rf.2f22c24b3e42c421545379ed08870c74.jpg",
        "2019-07-03_liposthey_leek_3_000081_jpg.rf.2125abbe5f7f8630443d07c38d4b24c0.jpg",
        "2019-07-03_liposthey_leek_2_000071_jpg.rf.4bc1fa842297e7225fc4574ef465b553.jpg",
        "2019-07-03_liposthey_leek_2_000046_jpg.rf.9e3facf231088bd81c81f0cb08295928.jpg",
        "2021-03-29_liposthey_leek_1_000006_jpg.rf.0e3cdeb347e30c853e2dacb008a14e25.jpg",
        "2021-03-29_liposthey_leek_1_000069_jpg.rf.b5243392afb8215708434dd5a9a73453.jpg",
        "2019-07-03_liposthey_leek_2_000081_jpg.rf.222830e486f2c03c99f8105bc960bc35.jpg",
        "2019-07-03_liposthey_leek_2_000103_jpg.rf.5aa9c5b1ed8334ee45469d7e3aec69dd.jpg",
        "2021-05-24_bordeaux_leek_3_000001_jpg.rf.c592a6cdc6cf8950e4f8887329933c87.jpg",
        "2021-03-29_liposthey_leek_1_000146_jpg.rf.bda78978cf1bbde478850ad1ac1b76df.jpg",
        "2019-07-03_liposthey_leek_3_000051_jpg.rf.0ca848fc745c8deb0faa5419642b1f98.jpg",
        "2021-03-29_liposthey_leek_1_000152_jpg.rf.6942f16394405dbad9ef885c1d06eab6.jpg",
        "2021-03-29_liposthey_leek_1_000136_jpg.rf.1141e3f55d194ea9d9af9d631be47b26.jpg",
        "2019-07-03_liposthey_leek_4_000017_jpg.rf.cec9198541f6e2c70417d3e3940794e1.jpg",
        "2021-03-29_liposthey_leek_1_000046_jpg.rf.7c6f521b499786a0f1d1d006bd97f9d5.jpg",
        "2019-07-03_liposthey_leek_2_000073_jpg.rf.900187323a87a4932379b89f2cd01505.jpg",
    ]

    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")


def main():
    config = VCDConfig()
    set_seed(config.SEED)
    try:
        json_path = config.DATA_ROOT / config.JSON_FILE
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        all_annotations, img_id_to_name = parse_coco_json_vcd(
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
