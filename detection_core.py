import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F_tv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import random
import numpy as np
import os
from PIL import Image
import gc
from tqdm import tqdm
import time
import wandb
import xml.etree.ElementTree as ET
from typing import Dict, Any
from torchmetrics.detection import MeanAveragePrecision
import traceback
import json


IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def detection_collate_fn(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id, seed=42):
    worker_seed = (seed + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def split_dataset_indices(image_ids, train_ratio=0.8, seed=42):
    sorted_ids = sorted(image_ids)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(sorted_ids))
    split_idx = int(len(sorted_ids) * train_ratio)
    train_ids = [sorted_ids[i] for i in shuffled_indices[:split_idx]]
    val_ids = [sorted_ids[i] for i in shuffled_indices[split_idx:]]
    return train_ids, val_ids


# ==================== PARSERS ====================

def parse_roboflow_xml(xml_path, class_filter='ridderzuring'):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            class_name = name_elem.text.lower()
            if class_filter in class_name:
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)
        return boxes, labels
    except Exception:
        return [], []


def parse_coco_json_sorghum(json_path):
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        image_annotations = {}
        image_id_to_name = {}
        for img_info in coco_data['images']:
            image_id_to_name[img_info['id']] = img_info['file_name']
        for img_id in image_id_to_name.keys():
            image_annotations[img_id] = {'boxes': [], 'labels': []}
        category_mapping = {1: 1, 2: 2, 3: 3}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']
            x, y, w, h = bbox
            if w > 0 and h > 0:
                x1, y1, x2, y2 = x, y, x + w, y + h
                if x2 > x1 and y2 > y1:
                    image_annotations[img_id]['boxes'].append([x1, y1, x2, y2])
                    category_id = ann['category_id']
                    image_annotations[img_id]['labels'].append(category_mapping.get(category_id, 0))
        return image_annotations, image_id_to_name
    except Exception:
        traceback.print_exc()
        return {}, {}


def parse_coco_json_thistle(json_path, test_images=None):
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        if test_images is None:
            test_images = []
        image_annotations = {}
        image_id_to_name = {}
        for img_info in coco_data['images']:
            if img_info['file_name'] in test_images:
                continue
            image_id_to_name[img_info['id']] = img_info['file_name']
        for img_id in image_id_to_name.keys():
            image_annotations[img_id] = {'boxes': [], 'labels': []}
        category_mapping = {0: 1}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_id_to_name:
                continue
            category_id = ann['category_id']
            if category_id not in category_mapping:
                continue
            bbox = ann['bbox']
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            if x2 > x1 and y2 > y1:
                image_annotations[img_id]['boxes'].append([x1, y1, x2, y2])
                image_annotations[img_id]['labels'].append(category_mapping[category_id])
        return image_annotations, image_id_to_name
    except Exception:
        traceback.print_exc()
        return {}, {}


def parse_coco_json_vcd(json_path, test_images=None):
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        image_annotations = {}
        image_id_to_name = {}
        test_images = test_images or []
        for img_info in coco_data['images']:
            file_name = img_info['file_name']
            if file_name in test_images:
                continue
            img_id = img_info['id']
            image_id_to_name[img_id] = file_name
        for img_id in image_id_to_name.keys():
            image_annotations[img_id] = {'boxes': [], 'labels': []}
        category_mapping = {0: 1}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_id_to_name:
                continue
            category_id = ann['category_id']
            if category_id not in category_mapping:
                continue
            bbox = ann['bbox']
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            if x2 > x1 and y2 > y1:
                image_annotations[img_id]['boxes'].append([x1, y1, x2, y2])
                image_annotations[img_id]['labels'].append(category_mapping[category_id])
        return image_annotations, image_id_to_name
    except Exception:
        traceback.print_exc()
        return {}, {}


# ==================== BACKBONE LOADING ====================

def load_backbone(config):
    if config.BACKBONE_TYPE == 'pretrained':
        if config.BACKBONE_ARCH == 'resnet50':
            backbone = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_channels = 2048
            return backbone
        elif config.BACKBONE_ARCH == 'convnext_tiny':
            backbone = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            backbone = backbone.features
            backbone.out_channels = 768
            return backbone
    elif config.BACKBONE_TYPE == 'wandb':
        return load_backbone_from_wandb(config)
    elif config.BACKBONE_TYPE == 'local':
        return load_backbone_from_local(config)
    raise ValueError(f"Invalid BACKBONE_TYPE: {config.BACKBONE_TYPE}")


def _load_and_clean_state_dict(checkpoint):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('base_model.', '', 1) if key.startswith('base_model.') else key
        new_state_dict[new_key] = value
    return new_state_dict


def _create_backbone_from_state_dict(state_dict, arch):
    if arch == 'resnet50':
        backbone = torchvision.models.resnet50(weights=None)
        backbone.load_state_dict(state_dict, strict=False)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        return backbone
    elif arch == 'convnext_tiny':
        full_model = torchvision.models.convnext_tiny(weights=None)
        full_model.load_state_dict(state_dict, strict=False)
        backbone = full_model.features
        backbone.out_channels = 768
        return backbone
    raise ValueError(f"Unsupported BACKBONE_ARCH: {arch}")


def load_backbone_from_wandb(config):
    try:
        temp_run = None
        if wandb.run is None:
            temp_run = wandb.init(
                project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY,
                notes="Temporary run for downloading backbone artifact", mode="online"
            )
        artifact = wandb.use_artifact(config.WANDB_ARTIFACT, type='model')
        artifact_dir = artifact.download()
        artifact_path = Path(artifact_dir)
        model_files = list(artifact_path.glob("*.pth")) + list(artifact_path.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No .pth or .pt files found in {artifact_dir}")
        checkpoint = torch.load(model_files[0], map_location='cpu')
        state_dict = _load_and_clean_state_dict(checkpoint)
        backbone = _create_backbone_from_state_dict(state_dict, config.BACKBONE_ARCH)
        if temp_run:
            wandb.finish()
        return backbone
    except Exception:
        traceback.print_exc()
        raise


def load_backbone_from_local(config):
    try:
        local_path = Path(config.LOCAL_BACKBONE_PATH)
        if not local_path.exists():
            raise FileNotFoundError(f"Local backbone file not found: {config.LOCAL_BACKBONE_PATH}")
        checkpoint = torch.load(config.LOCAL_BACKBONE_PATH, map_location='cpu')
        state_dict = _load_and_clean_state_dict(checkpoint)
        return _create_backbone_from_state_dict(state_dict, config.BACKBONE_ARCH)
    except Exception:
        traceback.print_exc()
        raise


# ==================== DATASETS ====================

class RoboflowDetectionDataset(Dataset):
    def __init__(self, data_dir, config, class_filter='ridderzuring', seed_offset=0):
        self.data_dir = Path(data_dir)
        self.config = config
        self.class_filter = class_filter
        self.image_paths = sorted(
            [p for p in self.data_dir.glob('*.jpg')],
            key=lambda x: x.name.lower()
        )
        if config.SUBSET_PERCENTAGE < 100:
            self._apply_subset_sampling(seed_offset)

    def _apply_subset_sampling(self, seed_offset):
        original_count = len(self.image_paths)
        keep_count = int(original_count * (self.config.SUBSET_PERCENTAGE / 100.0))
        if keep_count >= original_count:
            return
        rng = np.random.default_rng(self.config.SEED + seed_offset)
        indices = rng.choice(original_count, size=keep_count, replace=False)
        self.image_paths = [self.image_paths[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception:
            image = Image.new('RGB', (416, 416), color='black')
        xml_path = img_path.with_suffix('.xml')
        if not xml_path.exists():
            xml_path = Path(str(img_path).replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml'))
        if xml_path.exists():
            boxes, labels = parse_roboflow_xml(xml_path, self.class_filter)
        else:
            boxes, labels = [], []
        image_tensor = F_tv.to_tensor(image)
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        original_height, original_width = image_tensor.shape[1], image_tensor.shape[2]
        scale = min(self.config.MIN_SIZE / min(original_width, original_height),
                    self.config.MAX_SIZE / max(original_width, original_height))
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img_resized = F_tv.resize(image_tensor, [new_height, new_width])
        if len(boxes) > 0:
            resized_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                rx1, ry1, rx2, ry2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                if rx2 > rx1 and ry2 > ry1:
                    resized_boxes.append([rx1, ry1, rx2, ry2])
            if resized_boxes:
                boxes_tensor = torch.tensor(resized_boxes, dtype=torch.float32)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros(0, dtype=torch.int64)
        img_normalized = F_tv.normalize(img_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img_normalized, {'boxes': boxes_tensor, 'labels': labels_tensor}


class COCODetectionDataset(Dataset):
    def __init__(self, data_dir, config, annotations, img_id_to_name,
                 split_ids=None, seed_offset=0, image_folder=None, dummy_size=(2048, 1536)):
        self.data_dir = Path(data_dir)
        self.config = config
        self.annotations = annotations
        self.img_id_to_name = img_id_to_name
        self.image_folder = image_folder
        self.dummy_size = dummy_size
        self.image_paths = []
        self.image_ids = []
        for img_id, img_name in self.img_id_to_name.items():
            if split_ids is not None and img_id not in split_ids:
                continue
            if self.image_folder:
                img_path = Path(self.image_folder) / img_name
            else:
                img_path = self.data_dir / img_name
            if img_path.exists():
                self.image_paths.append(img_path)
                self.image_ids.append(img_id)
            else:
                img_stem = Path(img_name).stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    if self.image_folder:
                        alt_path = Path(self.image_folder) / (img_stem + ext)
                    else:
                        alt_path = self.data_dir / (img_stem + ext)
                    if alt_path.exists():
                        self.image_paths.append(alt_path)
                        self.image_ids.append(img_id)
                        break
        sorted_pairs = sorted(zip(self.image_paths, self.image_ids), key=lambda x: x[0].name.lower())
        if sorted_pairs:
            self.image_paths, self.image_ids = zip(*sorted_pairs)
            self.image_paths = list(self.image_paths)
            self.image_ids = list(self.image_ids)
        if config.SUBSET_PERCENTAGE < 100:
            self._apply_subset_sampling(seed_offset)

    def _apply_subset_sampling(self, seed_offset):
        original_count = len(self.image_paths)
        keep_count = int(original_count * (self.config.SUBSET_PERCENTAGE / 100.0))
        if keep_count >= original_count:
            return
        rng = np.random.default_rng(self.config.SEED + seed_offset)
        indices = rng.choice(original_count, size=keep_count, replace=False)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.image_ids = [self.image_ids[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = self.image_ids[idx]
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception:
            image = Image.new('RGB', self.dummy_size, color='black')
        if img_id in self.annotations:
            boxes = self.annotations[img_id]['boxes']
            labels = self.annotations[img_id]['labels']
        else:
            boxes, labels = [], []
        image_tensor = F_tv.to_tensor(image)
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        original_height, original_width = image_tensor.shape[1], image_tensor.shape[2]
        scale = min(self.config.MIN_SIZE / min(original_width, original_height),
                    self.config.MAX_SIZE / max(original_width, original_height))
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img_resized = F_tv.resize(image_tensor, [new_height, new_width])
        if len(boxes) > 0:
            resized_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                rx1, ry1, rx2, ry2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                if rx2 > rx1 and ry2 > ry1:
                    resized_boxes.append([rx1, ry1, rx2, ry2])
            if resized_boxes:
                boxes_tensor = torch.tensor(resized_boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros(0, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        img_normalized = F_tv.normalize(img_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img_normalized, {'boxes': boxes_tensor, 'labels': labels_tensor}


# ==================== MODEL ====================

def create_detection_model(config):
    backbone = load_backbone(config)
    anchor_generator = AnchorGenerator(sizes=config.ANCHOR_SIZES, aspect_ratios=config.ASPECT_RATIOS)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone, num_classes=config.NUM_CLASSES,
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
        min_size=config.MIN_SIZE, max_size=config.MAX_SIZE
    )
    if config.FREEZE_BACKBONE:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model


# ==================== COCO METRICS ====================

class COCOMetricsCalculator:

    @staticmethod
    def calculate_coco_metrics(predictions, targets, num_classes=2) -> Dict[str, Any]:
        try:
            predictions_cpu = [
                {'boxes': p['boxes'].detach().cpu(), 'scores': p['scores'].detach().cpu(), 'labels': p['labels'].detach().cpu()}
                for p in predictions
            ]
            targets_cpu = [
                {'boxes': t['boxes'].detach().cpu(), 'labels': t['labels'].detach().cpu()}
                for t in targets
            ]
            total_pred = sum(len(p['boxes']) for p in predictions_cpu)
            total_tgt = sum(len(t['boxes']) for t in targets_cpu)
            if total_pred == 0 or total_tgt == 0:
                return COCOMetricsCalculator._get_empty_coco_metrics(num_classes)
            metric = MeanAveragePrecision(
                box_format='xyxy', iou_thresholds=None, rec_thresholds=None,
                max_detection_thresholds=[1, 10, 100],
                class_metrics=(num_classes > 2), extended_summary=False
            )
            metric.update(predictions_cpu, targets_cpu)
            result = metric.compute()

            def safe_get(key, default=0.0):
                value = result.get(key)
                if value is None:
                    return default
                return value.item() if isinstance(value, torch.Tensor) else value

            coco_metrics = {
                'mAP': safe_get('map'), 'mAP_50': safe_get('map_50'), 'mAP_75': safe_get('map_75'),
                'mAP_small': safe_get('map_small'), 'mAP_medium': safe_get('map_medium'), 'mAP_large': safe_get('map_large'),
                'mar_1': safe_get('mar_1'), 'mar_10': safe_get('mar_10'), 'mar_100': safe_get('mar_100'),
                'mar_small': safe_get('mar_small'), 'mar_medium': safe_get('mar_medium'), 'mar_large': safe_get('mar_large'),
            }
            if 'map_per_class' in result:
                per_class = result['map_per_class']
                if per_class.dim() == 0:
                    coco_metrics['AP_class_1'] = 0.0 if per_class.item() == -1 else per_class.item()
                else:
                    for ci in range(min(num_classes, len(per_class))):
                        v = per_class[ci]
                        coco_metrics[f'AP_class_{ci}'] = 0.0 if v == -1 else (v.item() if isinstance(v, torch.Tensor) else float(v))
            return coco_metrics
        except Exception:
            traceback.print_exc()
            return COCOMetricsCalculator._get_empty_coco_metrics(num_classes)

    @staticmethod
    def _get_empty_coco_metrics(num_classes=2) -> Dict[str, Any]:
        metrics = {
            'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0,
            'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.0,
            'mar_1': 0.0, 'mar_10': 0.0, 'mar_100': 0.0,
            'mar_small': 0.0, 'mar_medium': 0.0, 'mar_large': 0.0,
        }
        for ci in range(num_classes):
            metrics[f'AP_class_{ci}'] = 0.0
        return metrics

    @staticmethod
    def log_coco_metrics_to_wandb(wandb_run, coco_metrics, epoch, class_names=None):
        if not wandb_run:
            return
        try:
            log_dict = {
                'epoch': epoch,
                'coco/mAP': coco_metrics['mAP'],
                'coco/mAP_50': coco_metrics['mAP_50'],
                'coco/mAP_75': coco_metrics['mAP_75'],
            }
            if class_names:
                for ci in range(1, len(class_names)):
                    ap_key = f'AP_class_{ci}'
                    if ap_key in coco_metrics:
                        log_dict[f'coco/AP_{class_names[ci]}'] = coco_metrics[ap_key]
            wandb_run.log(log_dict)
        except Exception:
            pass


# ==================== TRAINER ====================

class SimpleDetectionTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.coco_calculator = COCOMetricsCalculator()
        self.scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION and device.type == 'cuda' else None
        self.best_map = 0.0
        self.wandb = self._setup_wandb()

    def _setup_wandb(self):
        try:
            api_key = os.environ.get('WANDB_API_KEY')
            if not api_key:
                return None
            wandb.login(key=api_key)
            run = wandb.init(
                project=self.config.WANDB_PROJECT,
                entity=self.config.WANDB_ENTITY,
                name=self.config.WANDB_RUN_NAME,
                config={
                    'batch_size': self.config.BATCH_SIZE,
                    'learning_rate': self.config.LEARNING_RATE,
                    'epochs': self.config.NUM_EPOCHS,
                    'backbone_type': self.config.BACKBONE_TYPE,
                    'backbone_arch': self.config.BACKBONE_ARCH,
                    'classes': self.config.NUM_CLASSES,
                    'class_names': self.config.CLASS_NAMES,
                    'freeze_backbone': self.config.FREEZE_BACKBONE,
                    'subset_percentage': self.config.SUBSET_PERCENTAGE,
                    'wandb_artifact': self.config.WANDB_ARTIFACT if self.config.BACKBONE_TYPE == 'wandb' else None,
                    'environment': 'Kaggle' if IN_KAGGLE else 'Local'
                },
                tags=["detection", "faster-rcnn", self.config.BACKBONE_ARCH,
                      f"backbone-{self.config.BACKBONE_TYPE}",
                      f"subset-{self.config.SUBSET_PERCENTAGE}%",
                      "kaggle" if IN_KAGGLE else "local"]
            )
            return run
        except Exception:
            return None

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            valid_images = []
            valid_targets = []
            for i, target in enumerate(targets):
                if len(target['boxes']) > 0:
                    valid_images.append(images[i])
                    valid_targets.append(targets[i])
            if len(valid_images) == 0:
                continue
            images_dev = [img.to(self.device) for img in valid_images]
            targets_dev = [{'boxes': t['boxes'].to(self.device), 'labels': t['labels'].to(self.device)} for t in valid_targets]
            self.optimizer.zero_grad()
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    loss_dict = self.model(images_dev, targets_dev)
                    loss = sum(l for l in loss_dict.values())
                self.scaler.scale(loss).backward()
                if self.config.GRADIENT_CLIPPING > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images_dev, targets_dev)
                loss = sum(l for l in loss_dict.values())
                loss.backward()
                if self.config.GRADIENT_CLIPPING > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING)
                self.optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{running_loss / batch_count:.4f}'})
            if self.wandb and batch_idx % 10 == 0:
                self.wandb.log({
                    'batch_train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch, 'batch': batch_idx
                })
        return running_loss / batch_count if batch_count > 0 else 0.0

    def validate_epoch(self, epoch):
        self.model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc=f"Val {epoch}"):
                images_dev = [img.to(self.device) for img in images]
                predictions = self.model(images_dev)
                all_predictions.extend([
                    {'boxes': p['boxes'].cpu(), 'scores': p['scores'].cpu(), 'labels': p['labels'].cpu()}
                    for p in predictions
                ])
                all_targets.extend([
                    {'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()}
                    for t in targets
                ])
        coco_metrics = self.coco_calculator.calculate_coco_metrics(
            all_predictions, all_targets, self.config.NUM_CLASSES
        )
        if self.wandb:
            self.coco_calculator.log_coco_metrics_to_wandb(
                self.wandb, coco_metrics, epoch, self.config.CLASS_NAMES
            )
        return 0.0, coco_metrics

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            path = self.config.CHECKPOINT_DIR / f'best_model_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                'best_map': self.best_map,
                'config': {'num_classes': self.config.NUM_CLASSES, 'class_names': self.config.CLASS_NAMES}
            }, path, _use_new_zipfile_serialization=False)
        elif epoch % self.config.CHECKPOINT_SAVE_FREQUENCY == 0:
            path = self.config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            }, path, _use_new_zipfile_serialization=False)

    def save_final_model(self):
        if self.config.SAVE_FINAL_MODEL:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'backbone_type': self.config.BACKBONE_TYPE, 'backbone_arch': self.config.BACKBONE_ARCH,
                    'num_classes': self.config.NUM_CLASSES, 'class_names': self.config.CLASS_NAMES,
                    'min_size': self.config.MIN_SIZE, 'max_size': self.config.MAX_SIZE,
                    'subset_percentage': self.config.SUBSET_PERCENTAGE
                },
                'best_map': self.best_map
            }, self.config.FINAL_MODEL_PATH)

    def train(self, num_epochs):
        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, coco_metrics = self.validate_epoch(epoch)
            if self.scheduler:
                self.scheduler.step()
            current_map = coco_metrics['mAP']
            is_best = current_map > self.best_map
            if is_best:
                self.best_map = current_map
                self.save_checkpoint(epoch, is_best=True)
            if epoch % self.config.CHECKPOINT_SAVE_FREQUENCY == 0:
                self.save_checkpoint(epoch)
            if self.wandb:
                self.coco_calculator.log_coco_metrics_to_wandb(self.wandb, coco_metrics, epoch, self.config.CLASS_NAMES)
                self.wandb.log({'epoch': epoch, 'train/loss': train_loss, 'best/map': self.best_map})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        total_time = time.time() - start_time
        self.save_final_model()
        if self.wandb:
            summary_artifact = wandb.Artifact(name=f"training_summary_{self.config.WANDB_RUN_NAME}", type="metrics")
            summary_file = "final_training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    "best_map": self.best_map, "total_training_hours": total_time / 3600,
                    "final_epoch": num_epochs, "subset_percentage": self.config.SUBSET_PERCENTAGE,
                    "num_classes": self.config.NUM_CLASSES, "class_names": self.config.CLASS_NAMES,
                    "environment": "Kaggle" if IN_KAGGLE else "Local"
                }, f, indent=2)
            summary_artifact.add_file(summary_file)
            self.wandb.log_artifact(summary_artifact)
            self.wandb.finish()


# ==================== COMMON MAIN HELPERS ====================

def create_data_loaders(train_dataset, val_dataset, config):
    train_gen = torch.Generator()
    train_gen.manual_seed(config.SEED)
    val_gen = torch.Generator()
    val_gen.manual_seed(config.SEED + 1000)

    def _seed_worker(worker_id):
        seed_worker(worker_id, config.SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=detection_collate_fn,
        pin_memory=True, worker_init_fn=_seed_worker, generator=train_gen
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=detection_collate_fn,
        pin_memory=True, worker_init_fn=_seed_worker, generator=val_gen
    )
    return train_loader, val_loader


def run_detection_training(config, train_dataset, val_dataset):
    set_seed(config.SEED)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    model = create_detection_model(config)
    if config.FREEZE_BACKBONE:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    trainer = SimpleDetectionTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=config.DEVICE, config=config
    )
    trainer.train(config.NUM_EPOCHS)
