import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Train ViT on GPU with resume capability")
parser.add_argument('--resume', action='store_true', help="Resume from last checkpoint")
args = parser.parse_args()

NUM_EPOCHS = 53
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5

BATCH_SIZE = 32 if torch.cuda.is_available() else 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolarImageDataset(Dataset):
    def __init__(self, image_path, target_path, transform=None):
        self.images = np.load(image_path)
        targets_raw = np.load(target_path)
        self.targets = targets_raw.T
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        target = torch.from_numpy(self.targets[idx].astype(np.int64))
        return img, target

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
])

val_transform = None

script_dir = os.path.dirname(os.path.abspath(__file__))

train_dataset = PolarImageDataset(
    os.path.join(script_dir, 'Minput_images_polar_training.npy'),
    os.path.join(script_dir, 'Moutput_training.npy'),
    transform=train_transform
)

val_dataset = PolarImageDataset(
    os.path.join(script_dir, 'Minput_images_polar_test.npy'),
    os.path.join(script_dir, 'Moutput_test.npy'),
    transform=val_transform
)

pin_memory = DEVICE.type == 'cuda'
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except ImportError:
    import timm
    vit_model_fn = lambda: timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

class ViTMultiHead(nn.Module):
    def __init__(self, num_rings=5, num_classes_per_ring=11, pretrained=True, dropout_rate=0.5):
        super(ViTMultiHead, self).__init__()
        self.num_rings = num_rings
        self.num_classes_per_ring = num_classes_per_ring
        
        try:
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            vit = vit_b_16(weights=weights)
        except:
            import timm
            vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        
        if hasattr(vit, 'heads'):
            try:
                hidden_dim = vit.heads[0].in_features
            except Exception:
                hidden_dim = 768
            vit.heads = nn.Identity()
            self.backbone = vit
        else:
            self.backbone = vit
            hidden_dim = vit.embed_dim if hasattr(vit, 'embed_dim') else 768
        
        self.dropout = nn.Dropout(dropout_rate)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes_per_ring)
            for _ in range(num_rings)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        outputs = [head(features) for head in self.heads]
        return outputs

model = ViTMultiHead(num_rings=5, num_classes_per_ring=11, pretrained=False, dropout_rate=0.7).to(DEVICE)

checkpoint_path = os.path.join(script_dir, 'vit_last_checkpoint.pth')
start_epoch = 0
if args.resume and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    train_losses = ckpt.get('train_losses', [])
    val_losses = ckpt.get('val_losses', [])
else:
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

criterion = nn.CrossEntropyLoss()
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

scaler = torch.amp.GradScaler() if DEVICE.type == 'cuda' else None

if __name__ == '__main__':
    best_model_path = os.path.join(script_dir, 'best_vit_antenna_model.pth')

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_accuracy = [0.0] * 5
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=scaler is not None):
                outputs = model(images)
                loss = 0.0
                for ring_idx in range(5):
                    loss += criterion(outputs[ring_idx], targets[:, ring_idx])
                loss /= 5
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            
            for ring_idx in range(5):
                pred = outputs[ring_idx].argmax(dim=1)
                train_accuracy[ring_idx] += (pred == targets[:, ring_idx]).float().mean().item()
        
        train_loss /= len(train_loader)
        train_accuracy = [acc / len(train_loader) for acc in train_accuracy]
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_accuracy = [0.0] * 5
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                
                outputs = model(images)
                
                loss = 0.0
                for ring_idx in range(5):
                    loss += criterion(outputs[ring_idx], targets[:, ring_idx])
                loss /= 5
                
                val_loss += loss.item()
                
                for ring_idx in range(5):
                    pred = outputs[ring_idx].argmax(dim=1)
                    val_accuracy[ring_idx] += (pred == targets[:, ring_idx]).float().mean().item()
        
        val_loss /= len(val_loader)
        val_accuracy = [acc / len(val_loader) for acc in val_accuracy]
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {np.mean(train_accuracy):.4f}, Val Acc: {np.mean(val_accuracy):.4f}")
        print(f"  Per-ring Val Acc: {[f'{acc:.3f}' for acc in val_accuracy]}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        scheduler.step()
        
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), best_model_path)
            except Exception as e:
                print(f"Failed to save best model: {e}")
        
        print()

    end_time = time.time()
    print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (avg across rings)')
    plt.title('ViT Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'vit_training_loss.png'))
    print(f"\n✓ Loss curve saved to vit_training_loss.png")
    plt.show()

    print("\n✅ ViT training complete! Ready for evaluation.")
