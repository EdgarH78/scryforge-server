import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import xml.etree.ElementTree as ET
from scryforge.detector import Category  # Import Category enum
from torch.amp import autocast, GradScaler  # Update import

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# === Custom Dataset ===
class TokenDataset(Dataset):
    def __init__(self, root_dir, transforms=None, debug=False):
        self.root = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, "images")
        self.aug_img_dir = os.path.join(root_dir, "augmented/images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.aug_ann_dir = os.path.join(root_dir, "augmented/annotations")
        self.classes = Category.classes()

        # Combine regular and augmented data
        self.images = []
        self.annotations = []
        
        # Add regular data
        for img in sorted(os.listdir(self.img_dir)):
            if img.endswith('.jpg'):
                ann = img.replace('.jpg', '.xml')
                if os.path.exists(os.path.join(self.ann_dir, ann)):
                    self.images.append((self.img_dir, img))
                    self.annotations.append((self.ann_dir, ann))
                    if debug and len(self.images) >= 5:  # Only take 5 images in debug mode
                        break

        # Add augmented data
        for img in sorted(os.listdir(self.aug_img_dir)):
            if img.endswith('.jpg'):
                ann = img.replace('.jpg', '.xml')
                if os.path.exists(os.path.join(self.aug_ann_dir, ann)):
                    self.images.append((self.aug_img_dir, img))
                    self.annotations.append((self.aug_ann_dir, ann))
                    if debug and len(self.images) >= 5:  # Only take 5 images in debug mode
                        break

    def __getitem__(self, idx):
        img_dir, img_name = self.images[idx]
        ann_dir, ann_name = self.annotations[idx]

        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, ann_name)

        if not os.path.exists(img_path):
            print(f"❌ Missing image file: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("cv2.imread failed")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img)

            tree = ET.parse(ann_path)
            root = tree.getroot()

            boxes = []
            labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                label = self.classes.index(name)
                bndbox = obj.find("bndbox")
                box = [int(bndbox.find("xmin").text), int(bndbox.find("ymin").text),
                    int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)]
                boxes.append(box)
                labels.append(label)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
            return img_tensor, target

        except Exception as e:
            print(f"⚠️ Skipping {img_path} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.images)


# === Model Setup ===
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# === Training Loop ===
def collate_fn(batch):
    return tuple(zip(*batch))

def is_valid_target(t):
    if "boxes" not in t or not isinstance(t["boxes"], torch.Tensor):
        return False
    if t["boxes"].numel() == 0:
        return False
    if not torch.all(torch.isfinite(t["boxes"])):
        return False
    if t["boxes"].ndim != 2 or t["boxes"].shape[1] != 4:
        return False
    if torch.any((t["boxes"][:, 2:] - t["boxes"][:, :2]) <= 0):
        return False  # width or height <= 0
    return True


def train(debug=True):
    full_dataset = TokenDataset("dataset", debug=debug)
    print(f"Dataset size: {len(full_dataset)} images")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)  # 20% for validation
    train_size = total_size - val_size
    
    # Random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Set memory growth strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # More conservative batch sizes for 8GB VRAM
    train_loader = DataLoader(
        train_dataset, batch_size=12, shuffle=True,  # Reduced from 24
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,  # Reduced from 32
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2
    )

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_classes = len(Category.__members__) + 1  # All categories + background
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("⚠️ Warning: Training on CPU. This will be slow!")
    model = get_model(num_classes=num_classes)
    model.to(device)

    # Use mixed precision training
    scaler = GradScaler()
    
    # Optimize learning rate for larger batch size
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )

    # Optimize for CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # === Load checkpoint if it exists ===
    checkpoint_path = "fasterrcnn_token_detector.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_epoch = checkpoint["epoch"] + 1
                print(f"✅ Resumed training from epoch {start_epoch}")
            else:
                # Handle case where checkpoint is just the model state
                model.load_state_dict(checkpoint)
                print("✅ Loaded model weights from checkpoint")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("Starting fresh training...")

    num_epochs = 10
    if debug:
        num_epochs = 1
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Mixed precision training
            with autocast('cuda'):
                if not all(is_valid_target(t) for t in targets):
                    print(f"⚠️ Skipping batch with malformed targets")
                    continue
                loss_dict = model(images, targets)

                if isinstance(loss_dict, dict):
                    losses = sum(v for v in loss_dict.values() if isinstance(v, (int, float, torch.Tensor)))
                    if torch.isnan(losses):
                        print(f"🚨 NaN loss at batch {batch_idx}. Skipping.")
                        continue
                    train_loss += losses.item()

                    optimizer.zero_grad()
                    scaler.scale(losses).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print(f"⚠️ Skipping batch: model returned {type(loss_dict)}")
                    continue

                if batch_idx % 5 == 0:  # Print every 5 batches
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {losses.item():.4f}")

        # === Validation phase ===
        model.eval()
        total_detected = 0
        image_count = 0

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(val_loader):
                images = [img.to(device) for img in images]

                outputs = model(images)  # This returns a list of detections

                for i, output in enumerate(outputs):
                    num_boxes = output.get("boxes", torch.empty(0)).shape[0]
                    total_detected += num_boxes
                    image_count += 1

                if batch_idx % 5 == 0:
                    print(f"Validation | Batch {batch_idx}/{len(val_loader)}")

            print(f"🔍 Validation Summary: {total_detected} total boxes across {image_count} images")

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        # Save checkpoint with best validation loss
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, "fasterrcnn_token_detector.pth")

    scheduler.step(train_loss)
    print(f"Training on {train_size} samples, validating on {val_size} samples")

if __name__ == "__main__":
    train(debug=False)
