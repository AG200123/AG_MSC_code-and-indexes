import math
import os
import random
import time
import gc
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torchvision
import torchvision.models as models
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
from torchvision import transforms
import torchvision.transforms.functional as TF
import wandb
from tqdm import tqdm
import multiprocessing


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torchvision.set_image_backend('PIL')
    except:
        pass


def get_device():
    return config.DEVICE


def deterministic_worker_init_fn(worker_id):
    set_seed(config.SEED + worker_id)


def md5_split_indices(file_paths, labels, test_size=0.1):
    class_groups = defaultdict(list)
    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        class_groups[label].append((idx, file_path))
    train_indices = []
    val_indices = []
    for class_label, samples in class_groups.items():
        sorted_samples = sorted(samples, key=lambda x: hashlib.md5(Path(x[1]).name.encode()).hexdigest())
        indices = [idx for idx, _ in sorted_samples]
        seed_str = f"{config.SEED}_{class_label}"
        class_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2 ** 31)
        random.seed(class_seed)
        shuffled = indices[:]
        random.shuffle(shuffled)
        num_val = max(1, int(len(shuffled) * test_size))
        val_indices.extend(shuffled[:num_val])
        train_indices.extend(shuffled[num_val:])
    random.seed(config.SEED)
    random.shuffle(train_indices)
    random.seed(config.SEED + 12345)
    random.shuffle(val_indices)
    return train_indices, val_indices


def pad_image_to_size(image, model_input_size, padding_color=(0, 0, 0)):
    original_width, original_height = image.size
    target_width, target_height = model_input_size
    if original_width > target_width or original_height > target_height:
        image.thumbnail(model_input_size, Image.Resampling.LANCZOS)
        original_width, original_height = image.size
    if original_width < target_width or original_height < target_height:
        pad_left = (target_width - original_width) // 2
        pad_right = target_width - original_width - pad_left
        pad_top = (target_height - original_height) // 2
        pad_bottom = target_height - original_height - pad_top
        image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=padding_color)
    return image


def compute_class_weights(labels, idx_to_class, method='balanced', smoothing=1e-5, num_classes=None):
    if num_classes is None:
        num_classes = len(idx_to_class) if idx_to_class else len(set(labels))
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        if 0 <= label < num_classes:
            class_counts[label] += 1
    if method == 'inverse_frequency':
        weights = 1.0 / (class_counts + smoothing)
    elif method == 'balanced':
        weights = len(labels) / (num_classes * (class_counts + smoothing))
    elif method == 'sqrt_frequency':
        weights = 1.0 / torch.sqrt(class_counts + smoothing)
    elif method == 'effective_samples':
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + smoothing)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    weights = weights / weights.mean()
    max_ratio = config.MAX_WEIGHT_RATIO
    if weights.max() / weights.min() > max_ratio:
        max_allowed = weights.min() * max_ratio
        weights = torch.clamp(weights, max=max_allowed)
        weights = weights / weights.mean()
    return weights


def create_weighted_loss(weights, device):
    if weights is not None:
        weights = weights.to(device)
        weights = weights / weights.sum() * len(weights)
    return nn.CrossEntropyLoss(weight=weights)


class Config:
    DATA_ROOT = Path("")
    BATCH_SIZE = 128
    GRAD_ACCUM_STEPS = 1
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 8
    MIXED_PRECISION = True
    GRADIENT_CLIPPING = 1
    DATA_SUBSET_RATIO = 1
    VAL_RATIO = 0.1
    MIN_SAMPLES_PER_CLASS = 1
    SEED = 42
    MODEL_NAME = "resnet50"
    PRETRAINED = False
    PAD_SMALL_IMAGES = True
    MODEL_INPUT_SIZE = (224, 224)
    USE_ATTENTION_MASK = True
    ATTENTION_MASK_THRESHOLD = 1
    USE_WEIGHTED_LOSS = True
    WEIGHTED_LOSS_METHOD = 'inverse_frequency'
    WEIGHTED_LOSS_SMOOTHING = 1e-5
    MAX_WEIGHT_RATIO = 25
    LOG_PER_CLASS_METRICS = True
    WANDB_PROJECT = "x"
    WANDB_ENTITY = None
    WANDB_LOG_IMAGES = None
    WANDB_LOG_GRADIENTS = True
    WANDB_LOG_MODEL = True

    def __init__(self):
        self.CHECKPOINT_DIR = Path("checkpoints")
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.num_classes = None
        self.idx_to_class = None
        self.class_weights = None
        self.DEVICE = self._set_device()

    def _set_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("No CUDA available")

config = Config()


class JointSpatialTransform:
    def __init__(self, model_input_size, pad_small_images=True):
        self.model_input_size = model_input_size
        self.pad_small_images = pad_small_images

    def __call__(self, image, mask=None):
        if not self.pad_small_images:
            resize_size = (self.model_input_size[0] + 32, self.model_input_size[1] + 32)
            image = TF.resize(image, resize_size)
            if mask:
                mask = TF.resize(mask, resize_size, interpolation=transforms.InterpolationMode.NEAREST)
            i, j, h, w = transforms.RandomCrop.get_params(image, self.model_input_size)
            image = TF.crop(image, i, j, h, w)
            if mask:
                mask = TF.crop(mask, i, j, h, w)
        if random.random() < 0.3:
            image = TF.hflip(image)
            if mask:
                mask = TF.hflip(mask)
        angle = random.uniform(-5, 5)
        image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        if mask:
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask


class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, spatial_transform=None, epoch=0,
                 pad_small_images=False, model_input_size=(224, 224), use_attention_mask=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.spatial_transform = spatial_transform
        self.epoch = epoch
        self.pad_small_images = pad_small_images
        self.model_input_size = model_input_size
        self.use_attention_mask = use_attention_mask
        self._error_counts = {}
        self.dummy_image = Image.new('RGB', self.model_input_size, color=(0, 0, 0))

    def __len__(self):
        return len(self.file_paths)

    def safe_open_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img.verify()
                img = Image.open(img_path).convert('RGB')
                return img
        except (IOError, OSError, Image.DecompressionBombError):
            try:
                with Image.open(img_path) as img:
                    img.load()
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return img
            except Exception:
                return self.dummy_image.copy()
        except Exception:
            return self.dummy_image.copy()

    def __getitem__(self, idx):
        try:
            img_path = self.file_paths[idx]
            label = self.labels[idx]
            seed = config.SEED + self.epoch + idx
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            image = self.safe_open_image(img_path)
            if self.pad_small_images:
                image = pad_image_to_size(image, self.model_input_size, padding_color=(0, 0, 0))
            if self.use_attention_mask:
                try:
                    img_array = np.array(image)
                    mask = np.any(img_array > config.ATTENTION_MASK_THRESHOLD, axis=2).astype(np.float32)
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                except:
                    mask_pil = Image.fromarray(np.ones(self.model_input_size[::-1], dtype=np.uint8) * 255)
            else:
                mask_pil = None
            joint_transform = JointSpatialTransform(self.model_input_size, self.pad_small_images)
            if self.use_attention_mask and mask_pil is not None:
                image, mask_pil = joint_transform(image, mask_pil)
            else:
                image, _ = joint_transform(image, None)
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception:
                    image = self.transform(self.dummy_image.copy())
            if self.use_attention_mask and mask_pil is not None:
                try:
                    mask_array = np.array(mask_pil)
                    attention_mask = torch.from_numpy(mask_array).float() / 255.0
                    attention_mask = attention_mask.unsqueeze(0)
                except:
                    attention_mask = torch.ones(1, *self.model_input_size[::-1])
            if self.use_attention_mask:
                return image, attention_mask, label
            else:
                return image, label
        except Exception:
            dummy_image = Image.new('RGB', self.model_input_size, color=(0, 0, 0))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            safe_label = self.labels[idx] if idx < len(self.labels) else 0
            if self.use_attention_mask:
                return dummy_image, torch.ones(1, *self.model_input_size[::-1]), safe_label
            return dummy_image, safe_label

    def set_epoch(self, epoch):
        self.epoch = epoch


class DatasetManager:
    def __init__(self, data_root, seed=42, data_subset_ratio=0.1, min_samples_per_class=50):
        self.data_root = Path(data_root)
        self.seed = seed
        self.data_subset_ratio = data_subset_ratio
        self.min_samples_per_class = min_samples_per_class
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.train_labels = None

    def compute_image_hash(self, image_path):
        try:
            return hashlib.md5(Path(image_path).name.encode()).hexdigest()
        except Exception:
            return None

    def compute_train_class_counts(self, train_labels):
        train_class_counts = [0] * len(self.class_to_idx)
        for label in train_labels:
            train_class_counts[label] += 1
        return train_class_counts

    def discover_data(self):
        if not self.data_root.exists():
            return [], []
        all_image_paths = []
        all_labels = []
        filename_hashes = set()
        class_counts = defaultdict(int)
        class_folders = []
        has_nested_structure = False
        for folder in self.data_root.iterdir():
            if folder.is_dir():
                subfolders = [f for f in folder.iterdir() if f.is_dir()]
                if subfolders:
                    has_nested_structure = True
                    for class_folder in subfolders:
                        class_folders.append(class_folder)
        if not has_nested_structure:
            for folder in self.data_root.iterdir():
                if folder.is_dir():
                    class_folders.append(folder)
        class_folders.sort(key=lambda x: x.name)
        if not class_folders:
            return [], []
        unique_classes = sorted(set([cf.name for cf in class_folders]))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        for class_folder in class_folders:
            class_name = class_folder.name
            class_idx = self.class_to_idx[class_name]
            images_in_class = 0
            class_image_paths = []
            all_images = []
            for img_path in class_folder.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp']:
                    all_images.append(img_path)
            all_images.sort(key=lambda x: self.compute_image_hash(x) or str(x))
            for img_path in all_images:
                filename_hash = self.compute_image_hash(img_path)
                if filename_hash and filename_hash not in filename_hashes:
                    filename_hashes.add(filename_hash)
                    class_image_paths.append((str(img_path), filename_hash))
                    images_in_class += 1
            if images_in_class < self.min_samples_per_class:
                continue
            samples_to_use = max(1, math.floor(images_in_class * self.data_subset_ratio))
            random.seed(self.seed + class_idx)
            selected_samples = random.sample(class_image_paths, samples_to_use)
            for img_path, _ in selected_samples:
                all_image_paths.append(img_path)
                all_labels.append(class_idx)
                class_counts[class_idx] += 1
        return all_image_paths, all_labels

    def create_deterministic_transforms(self):
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform

    def create_md5_split_data_loaders(self, batch_size=16, num_workers=4, test_size=0.1):
        train_transform, val_transform = self.create_deterministic_transforms()
        all_image_paths, all_labels = self.discover_data()
        if len(all_image_paths) == 0:
            return None, None, None
        train_indices, val_indices = md5_split_indices(all_image_paths, all_labels, test_size)
        train_paths = [all_image_paths[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_paths = [all_image_paths[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        self.train_labels = train_labels
        if config.USE_WEIGHTED_LOSS:
            config.class_weights = compute_class_weights(
                labels=train_labels, idx_to_class=self.idx_to_class,
                method=config.WEIGHTED_LOSS_METHOD, smoothing=config.WEIGHTED_LOSS_SMOOTHING,
                num_classes=len(self.class_to_idx)
            )
        else:
            config.class_weights = None
        config.train_class_counts = self.compute_train_class_counts(train_labels)
        del all_image_paths, all_labels
        gc.collect()
        train_dataset = ImageDataset(
            train_paths, train_labels, train_transform, epoch=0,
            pad_small_images=config.PAD_SMALL_IMAGES, model_input_size=config.MODEL_INPUT_SIZE,
            use_attention_mask=config.USE_ATTENTION_MASK
        )
        val_dataset = ImageDataset(
            val_paths, val_labels, val_transform, epoch=0,
            pad_small_images=config.PAD_SMALL_IMAGES, model_input_size=config.MODEL_INPUT_SIZE,
            use_attention_mask=config.USE_ATTENTION_MASK
        )
        train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(config.SEED))
        val_sampler = SequentialSampler(val_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, persistent_workers=False,
            prefetch_factor=8, drop_last=True, worker_init_fn=deterministic_worker_init_fn, timeout=3000
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True, persistent_workers=False,
            prefetch_factor=8, drop_last=False, worker_init_fn=deterministic_worker_init_fn, timeout=3000
        )
        return train_loader, val_loader, train_dataset


def init_weights_deterministic(m):
    if not hasattr(init_weights_deterministic, 'counter'):
        init_weights_deterministic.counter = 0
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        torch.manual_seed(config.SEED + init_weights_deterministic.counter)
        init_weights_deterministic.counter += 1
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.manual_seed(config.SEED + init_weights_deterministic.counter)
                nn.init.constant_(m.bias, 0)
                init_weights_deterministic.counter += 1
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            torch.manual_seed(config.SEED + init_weights_deterministic.counter)
            nn.init.constant_(m.bias, 0)
            init_weights_deterministic.counter += 1
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.manual_seed(config.SEED + init_weights_deterministic.counter)
                nn.init.constant_(m.bias, 0)
                init_weights_deterministic.counter += 1


class Model(nn.Module):
    def __init__(self, base_model, num_classes, use_attention_mask=True):
        super().__init__()
        self.base_model = base_model
        self.use_attention_mask = use_attention_mask
        model_name = config.MODEL_NAME.lower()
        if 'convnext' in model_name:
            in_features = base_model.classifier[2].in_features
            base_model.classifier[2] = nn.Identity()
        else:
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, attention_mask=None):
        if self.use_attention_mask and attention_mask is not None:
            if attention_mask.shape[-2:] != x.shape[-2:]:
                attention_mask = F.interpolate(attention_mask, size=x.shape[-2:], mode='nearest')
            x = x * attention_mask
        features = self.base_model(x)
        return self.classifier(features)


def create_deterministic_model(num_classes, pretrained=False):
    set_seed(config.SEED)
    model_name = config.MODEL_NAME.lower()
    model_registry = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
        "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
        "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "convnext_small": (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
        "convnext_base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
        "convnext_large": (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
    }
    if model_name not in model_registry:
        raise ValueError(f"Unknown model name: {model_name}")
    model_fn, weight_class = model_registry[model_name]
    base_model = model_fn(weights=weight_class if pretrained else None)
    model = Model(base_model, num_classes, use_attention_mask=config.USE_ATTENTION_MASK)
    if not pretrained:
        model.apply(init_weights_deterministic)
    return model


def create_deterministic_optimizer(model, learning_rate=0.001, weight_decay=1e-4):
    set_seed(config.SEED)
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_deterministic_scheduler(optimizer, num_epochs):
    set_seed(config.SEED)
    if num_epochs <= 1:
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)


class Trainer:
    def __init__(self, model, train_loader, val_loader, train_dataset, optimizer, scheduler, device, checkpoint_dir, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.scaler = torch.amp.GradScaler('cuda') if (config.MIXED_PRECISION and device.type == 'cuda') else None
        if config.USE_WEIGHTED_LOSS and config.class_weights is not None:
            self.criterion = create_weighted_loss(config.class_weights, device)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.epoch_failures = 0
        self.max_epoch_failures = 3
        self.wandb = setup_wandb(config)
        if self.wandb:
            self.wandb.watch(model, log="all" if config.WANDB_LOG_GRADIENTS else "parameters", log_freq=100)

    def train_epoch(self, epoch):
        self.train_dataset.set_epoch(epoch)
        self.model.train()
        set_seed(config.SEED + epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        accumulation_steps = self.config.GRAD_ACCUM_STEPS
        self.optimizer.zero_grad()
        batch_skip_count = 0
        max_skip_batches = 100
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch_data in enumerate(pbar):
            try:
                if self.config.USE_ATTENTION_MASK:
                    data, attention_masks, targets = batch_data
                    data, targets = data.to(self.device), targets.to(self.device)
                    attention_masks = attention_masks.to(self.device)
                else:
                    data, targets = batch_data
                    data, targets = data.to(self.device), targets.to(self.device)
                    attention_masks = None
                if torch.isnan(data).any() or torch.isinf(data).any():
                    batch_skip_count += 1
                    if batch_skip_count > max_skip_batches:
                        raise RuntimeError(f"Too many skipped batches in epoch {epoch}")
                    continue
                if self.scaler and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(data, attention_masks)
                        loss = self.criterion(outputs, targets) / accumulation_steps
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.optimizer.zero_grad()
                        batch_skip_count += 1
                        continue
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(data, attention_masks)
                    loss = self.criterion(outputs, targets) / accumulation_steps
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.optimizer.zero_grad()
                        batch_skip_count += 1
                        continue
                    loss.backward()
                running_loss += loss.item() * accumulation_steps
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.wandb and batch_idx % 100 == 0:
                    self.wandb.log({
                        'batch_train_loss': loss.item() * accumulation_steps,
                        'batch_accuracy': 100. * correct / total if total > 0 else 0,
                        'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.LEARNING_RATE,
                        'epoch': epoch, 'batch': batch_idx
                    })
                if batch_idx % 200 == 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            except RuntimeError as e:
                batch_skip_count += 1
                if "CUDA out of memory" in str(e) and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                if batch_skip_count > max_skip_batches:
                    raise RuntimeError(f"Too many errors in epoch {epoch}")
                continue
            except Exception:
                batch_skip_count += 1
                if batch_skip_count > max_skip_batches:
                    raise RuntimeError(f"Too many errors in epoch {epoch}")
        if len(self.train_loader) % accumulation_steps != 0:
            try:
                if self.scaler and self.device.type == 'cuda':
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING)
                    self.optimizer.step()
                self.optimizer.zero_grad()
            except Exception:
                pass
        valid_batches = len(self.train_loader) - batch_skip_count
        epoch_loss = running_loss / valid_batches if valid_batches > 0 else 0.0
        epoch_acc = 100. * correct / total if total > 0 else 0.0
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return epoch_loss, epoch_acc, batch_skip_count

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_skip_count = 0
        per_class_correct = torch.zeros(self.config.num_classes, device=self.device)
        per_class_total = torch.zeros(self.config.num_classes, device=self.device)
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc=f"Val {epoch}"):
                try:
                    if self.config.USE_ATTENTION_MASK:
                        data, attention_masks, targets = batch_data
                        data, targets = data.to(self.device), targets.to(self.device)
                        attention_masks = attention_masks.to(self.device)
                    else:
                        data, targets = batch_data
                        data, targets = data.to(self.device), targets.to(self.device)
                        attention_masks = None
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        batch_skip_count += 1
                        continue
                    if self.scaler and self.device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(data, attention_masks)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(data, attention_masks)
                        loss = self.criterion(outputs, targets)
                    if torch.isnan(loss) or torch.isinf(loss):
                        batch_skip_count += 1
                        continue
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    for t, p in zip(targets, predicted):
                        per_class_total[t] += 1
                        if t == p:
                            per_class_correct[t] += 1
                except Exception:
                    batch_skip_count += 1
                    continue
        valid_batches = len(self.val_loader) - batch_skip_count
        epoch_loss = running_loss / valid_batches if valid_batches > 0 else 0.0
        epoch_acc = 100. * correct / total if total > 0 else 0.0
        if self.config.LOG_PER_CLASS_METRICS and total > 0:
            per_class_accuracy = per_class_correct / (per_class_total + 1e-8)
            class_sizes = torch.tensor([self.config.train_class_counts[i] for i in range(self.config.num_classes)], device=self.device)
            small_classes = class_sizes < 100
            large_classes = class_sizes >= 1000
            if self.wandb:
                small_acc = per_class_accuracy[small_classes].mean().item() * 100 if small_classes.any() else 0
                large_acc = per_class_accuracy[large_classes].mean().item() * 100 if large_classes.any() else 0
                self.wandb.log({
                    'small_class_accuracy': small_acc,
                    'large_class_accuracy': large_acc,
                    'accuracy_disparity': abs(large_acc - small_acc) if (small_classes.any() and large_classes.any()) else 0
                })
        return epoch_loss, epoch_acc, batch_skip_count

    def save_checkpoint(self, epoch, is_best=False):
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_losses': self.train_losses, 'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies, 'val_accuracies': self.val_accuracies,
                'best_val_acc': self.best_val_acc,
                'config': {
                    'SEED': self.config.SEED, 'MODEL_NAME': self.config.MODEL_NAME,
                    'num_classes': self.config.num_classes, 'PAD_SMALL_IMAGES': self.config.PAD_SMALL_IMAGES,
                    'MODEL_INPUT_SIZE': self.config.MODEL_INPUT_SIZE,
                    'USE_WEIGHTED_LOSS': self.config.USE_WEIGHTED_LOSS,
                    'WEIGHTED_LOSS_METHOD': self.config.WEIGHTED_LOSS_METHOD,
                    'USE_ATTENTION_MASK': self.config.USE_ATTENTION_MASK,
                    'ATTENTION_MASK_THRESHOLD': self.config.ATTENTION_MASK_THRESHOLD,
                    'class_weights': self.config.class_weights.tolist() if self.config.class_weights is not None else None,
                }
            }
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            if epoch > 5:
                old = self.checkpoint_dir / f'checkpoint_epoch_{epoch - 5}.pth'
                if old.exists():
                    old.unlink()
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                if self.wandb and self.config.WANDB_LOG_MODEL:
                    try:
                        artifact = wandb.Artifact(f"best-model-{self.wandb.id}", type="model",
                                                  description=f"Best model val_acc_{self.best_val_acc:.2f}%")
                        artifact.add_file(str(best_path))
                        self.wandb.log_artifact(artifact)
                    except Exception:
                        pass
        except Exception:
            pass

    def train(self, num_epochs):
        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            self.epoch_failures = 0
            try:
                set_seed(self.config.SEED + epoch)
                epoch_start = time.time()
                train_loss, train_acc, train_skipped = self.train_epoch(epoch)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                val_loss, val_acc, val_skipped = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                if self.scheduler:
                    self.scheduler.step()
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best)
                if self.wandb:
                    try:
                        self.wandb.log({
                            'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                            'val_loss': val_loss, 'val_accuracy': val_acc,
                            'best_val_accuracy': self.best_val_acc,
                            'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.LEARNING_RATE,
                            'epoch_time': time.time() - epoch_start,
                            'train_batches_skipped': train_skipped, 'val_batches_skipped': val_skipped,
                            'gpu_memory_used': torch.cuda.memory_allocated() / 1024 ** 3 if self.device.type == 'cuda' else 0,
                        })
                    except Exception:
                        pass
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                self.epoch_failures += 1
                if "DataLoader worker" in str(e) or "CUDA out of memory" in str(e):
                    if self.epoch_failures >= self.max_epoch_failures:
                        break
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                raise
            except Exception:
                self.epoch_failures += 1
                if self.epoch_failures >= self.max_epoch_failures:
                    break
                continue
        try:
            torch.save({
                'epoch': len(self.train_losses), 'model_state_dict': self.model.state_dict(),
                'best_val_acc': self.best_val_acc, 'config': self.config.__dict__,
            }, self.checkpoint_dir / 'final_model.pth')
        except Exception:
            pass
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:
                pass


def setup_wandb(cfg):
    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        return None
    try:
        wandb.login(key=api_key)
        config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
        config_dict.update({k: v for k, v in Config.__dict__.items()
                            if not k.startswith('_') and not callable(getattr(Config, k))})
        return wandb.init(
            project=cfg.WANDB_PROJECT, entity=cfg.WANDB_ENTITY, config=config_dict,
            notes=f"Classification with {cfg.MODEL_NAME}", tags=["classification", cfg.MODEL_NAME]
        )
    except Exception:
        return None


def main():
    set_seed(config.SEED)
    device = get_device()
    try:
        dataset_manager = DatasetManager(
            config.DATA_ROOT, seed=config.SEED,
            data_subset_ratio=config.DATA_SUBSET_RATIO,
            min_samples_per_class=config.MIN_SAMPLES_PER_CLASS
        )
        train_loader, val_loader, train_dataset = dataset_manager.create_md5_split_data_loaders(
            batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, test_size=config.VAL_RATIO
        )
        if train_loader is None or val_loader is None:
            return
        config.num_classes = len(dataset_manager.class_to_idx)
        config.idx_to_class = dataset_manager.idx_to_class
        if config.num_classes == 0:
            return
        model = create_deterministic_model(num_classes=config.num_classes, pretrained=config.PRETRAINED)
        optimizer = create_deterministic_optimizer(model, learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = create_deterministic_scheduler(optimizer, config.NUM_EPOCHS)
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            train_dataset=train_dataset, optimizer=optimizer, scheduler=scheduler,
            device=device, checkpoint_dir=config.CHECKPOINT_DIR, config=config
        )
        trainer.train(config.NUM_EPOCHS)
    except Exception:
        import traceback
        traceback.print_exc()
        if 'trainer' in locals() and hasattr(trainer, 'wandb') and trainer.wandb:
            trainer.wandb.finish()

if __name__ == "__main__":
    main()
