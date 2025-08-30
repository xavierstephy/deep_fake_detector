<<<<<<< HEAD
import os
import sys
import time
import json
import copy
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def face_crop(image_bgr: np.ndarray, detector: cv2.CascadeClassifier, margin: float = 0.2) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return image_bgr
    # Choose the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(margin * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_bgr.shape[1], x + w + pad)
    y1 = min(image_bgr.shape[0], y + h + pad)
    return image_bgr[y0:y1, x0:x1]


class ImageFolderWithFaceCrop(datasets.ImageFolder):
    def __init__(self, root: str, transform=None, face_crop_enabled: bool = False):
        super().__init__(root, transform=transform)
        self.face_crop_enabled = face_crop_enabled
        self._detector = load_face_detector() if face_crop_enabled else None

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = cv2.imread(path, cv2.IMREAD_COLOR)
        if sample is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if self.face_crop_enabled and self._detector is not None:
            sample = face_crop(sample, self._detector)
        # Convert BGR->RGB and to PIL-like array
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        # transforms in torchvision expect PIL or ndarray (HWC) with ToTensor
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def build_dataloaders(data_dir: str, img_size: int, batch_size: int, face_crop_enabled: bool, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, int]:
    train_dir = os.path.join(data_dir, "train")
    
    # Check if val folder exists, if not create split from train
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        print(f"Validation folder not found. Creating {val_split*100:.0f}% validation split from training data...")
        create_validation_split(train_dir, val_dir, val_split)

    train_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolderWithFaceCrop(train_dir, transform=train_tfms, face_crop_enabled=face_crop_enabled)
    val_ds = ImageFolderWithFaceCrop(val_dir, transform=val_tfms, face_crop_enabled=face_crop_enabled)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, num_classes


def create_validation_split(train_dir: str, val_dir: str, val_split: float = 0.2):
    """Create validation split from training data"""
    import random
    import shutil
    
    random.seed(42)  # For reproducibility
    
    # Create validation directory
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all class folders
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for class_name in classes:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        # Create validation class directory
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all files in the class
        files = [f for f in os.listdir(train_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # Randomly select files for validation
        num_val = int(len(files) * val_split)
        val_files = random.sample(files, num_val)
        
        # Move files to validation
        for file_name in val_files:
            src = os.path.join(train_class_dir, file_name)
            dst = os.path.join(val_class_dir, file_name)
            shutil.move(src, dst)
        
        print(f"Class {class_name}: {len(files)} total, {len(val_files)} moved to validation")


def build_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds = []
    all_targets = []
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.softmax(outputs.detach(), dim=1)[:, 1] if outputs.shape[1] == 2 else torch.argmax(outputs.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    try:
        if all_preds.ndim == 1:  # probabilities for class 1
            auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
        else:
            auc = roc_auc_score(all_targets.numpy(), all_preds[:, 1].numpy())
    except Exception:
        auc = float('nan')
    acc = accuracy_score(all_targets.numpy(), (all_preds >= 0.5).numpy() if all_preds.ndim == 1 else all_preds.numpy())
    return float(np.mean(losses)), float(acc), float(auc)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Val", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            preds = torch.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    try:
        if all_preds.ndim == 1:
            auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
        else:
            auc = roc_auc_score(all_targets.numpy(), all_preds[:, 1].numpy())
    except Exception:
        auc = float('nan')
    acc = accuracy_score(all_targets.numpy(), (all_preds >= 0.5).numpy() if all_preds.ndim == 1 else all_preds.numpy())
    return float(np.mean(losses)), float(acc), float(auc)


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def export_torchscript(model: nn.Module, img_size: int, output_path: str, device: torch.device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, output_path)


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector (ImageFolder)")
    parser.add_argument("--data-dir", required=True, help="Root with train/ and val/ folders")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--face-crop", action="store_true", help="Enable OpenCV face cropping before transforms")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    train_loader, val_loader, num_classes = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.face_crop, val_split=0.2
    )

    model = build_model(num_classes, model_name=args.model, pretrained=(not args.no_pretrained))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f} auc={train_auc:.4f}")
        print(f"  Val  : loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
        })

        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))

    # Save history
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best and export TorchScript
    if best_state is not None:
        model.load_state_dict(best_state)
    export_torchscript(model, args.img_size, os.path.join(args.out_dir, "model_ts.pt"), device)
    print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()


=======
import os
import sys
import time
import json
import copy
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def face_crop(image_bgr: np.ndarray, detector: cv2.CascadeClassifier, margin: float = 0.2) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return image_bgr
    # Choose the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(margin * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_bgr.shape[1], x + w + pad)
    y1 = min(image_bgr.shape[0], y + h + pad)
    return image_bgr[y0:y1, x0:x1]


class ImageFolderWithFaceCrop(datasets.ImageFolder):
    def __init__(self, root: str, transform=None, face_crop_enabled: bool = False):
        super().__init__(root, transform=transform)
        self.face_crop_enabled = face_crop_enabled
        self._detector = load_face_detector() if face_crop_enabled else None

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = cv2.imread(path, cv2.IMREAD_COLOR)
        if sample is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if self.face_crop_enabled and self._detector is not None:
            sample = face_crop(sample, self._detector)
        # Convert BGR->RGB and to PIL-like array
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        # transforms in torchvision expect PIL or ndarray (HWC) with ToTensor
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def build_dataloaders(data_dir: str, img_size: int, batch_size: int, face_crop_enabled: bool, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, int]:
    train_dir = os.path.join(data_dir, "train")
    
    # Check if val folder exists, if not create split from train
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        print(f"Validation folder not found. Creating {val_split*100:.0f}% validation split from training data...")
        create_validation_split(train_dir, val_dir, val_split)

    train_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolderWithFaceCrop(train_dir, transform=train_tfms, face_crop_enabled=face_crop_enabled)
    val_ds = ImageFolderWithFaceCrop(val_dir, transform=val_tfms, face_crop_enabled=face_crop_enabled)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, num_classes


def create_validation_split(train_dir: str, val_dir: str, val_split: float = 0.2):
    """Create validation split from training data"""
    import random
    import shutil
    
    random.seed(42)  # For reproducibility
    
    # Create validation directory
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all class folders
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for class_name in classes:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        # Create validation class directory
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all files in the class
        files = [f for f in os.listdir(train_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # Randomly select files for validation
        num_val = int(len(files) * val_split)
        val_files = random.sample(files, num_val)
        
        # Move files to validation
        for file_name in val_files:
            src = os.path.join(train_class_dir, file_name)
            dst = os.path.join(val_class_dir, file_name)
            shutil.move(src, dst)
        
        print(f"Class {class_name}: {len(files)} total, {len(val_files)} moved to validation")


def build_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds = []
    all_targets = []
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.softmax(outputs.detach(), dim=1)[:, 1] if outputs.shape[1] == 2 else torch.argmax(outputs.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    try:
        if all_preds.ndim == 1:  # probabilities for class 1
            auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
        else:
            auc = roc_auc_score(all_targets.numpy(), all_preds[:, 1].numpy())
    except Exception:
        auc = float('nan')
    acc = accuracy_score(all_targets.numpy(), (all_preds >= 0.5).numpy() if all_preds.ndim == 1 else all_preds.numpy())
    return float(np.mean(losses)), float(acc), float(auc)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Val", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            preds = torch.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    try:
        if all_preds.ndim == 1:
            auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
        else:
            auc = roc_auc_score(all_targets.numpy(), all_preds[:, 1].numpy())
    except Exception:
        auc = float('nan')
    acc = accuracy_score(all_targets.numpy(), (all_preds >= 0.5).numpy() if all_preds.ndim == 1 else all_preds.numpy())
    return float(np.mean(losses)), float(acc), float(auc)


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def export_torchscript(model: nn.Module, img_size: int, output_path: str, device: torch.device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, output_path)


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector (ImageFolder)")
    parser.add_argument("--data-dir", required=True, help="Root with train/ and val/ folders")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--face-crop", action="store_true", help="Enable OpenCV face cropping before transforms")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    train_loader, val_loader, num_classes = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.face_crop, val_split=0.2
    )

    model = build_model(num_classes, model_name=args.model, pretrained=(not args.no_pretrained))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f} auc={train_auc:.4f}")
        print(f"  Val  : loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
        })

        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "num_classes": num_classes,
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))

    # Save history
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best and export TorchScript
    if best_state is not None:
        model.load_state_dict(best_state)
    export_torchscript(model, args.img_size, os.path.join(args.out_dir, "model_ts.pt"), device)
    print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()


>>>>>>> 0c57338d10119562221af77a603da511f81f8170
