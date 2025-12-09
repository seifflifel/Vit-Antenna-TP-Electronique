import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime


### Affichage visuel de démarcation
def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()

# === Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    BATCH_SIZE = 16  # RTX 3050 can handle this well (~6GB VRAM)
else:
    print("⚠ CUDA not available. Using CPU (slower training).")
    print("  To use GPU, update NVIDIA drivers and reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    BATCH_SIZE = 4  # Reduce batch size for CPU

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 5

# For testing on CPU, reduce epochs:
if not torch.cuda.is_available():
    print("⚠ Note: Running on CPU. Consider reducing NUM_EPOCHS for testing.")
    # NUM_EPOCHS = 3  # Uncomment this line for quick CPU test

# === Dataset ===
class PolarImageDataset(Dataset):
    def __init__(self, image_path, target_path, transform=None):
        """
        Load polar images and corresponding architecture targets.
        Images: (N, 240, 240, 3) uint8
        Targets: (5, N) → targets per ring (0-10 element counts)
        """
        self.images = np.load(image_path)  # (N, H, W, 3)
        targets_raw = np.load(target_path)  # (5, N)
        self.targets = targets_raw.T  # transpose to (N, 5)
        self.transform = transform
        
        print(f"Dataset loaded: {len(self)} images")
        print(f"  Images shape: {self.images.shape}")
        print(f"  Targets shape: {self.targets.shape}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Image: (240, 240, 3) uint8 → (3, 240, 240) float32 [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = torch.from_numpy(img)
        
        # Target: architecture counts (5 values, each 0-10)
        target = torch.from_numpy(self.targets[idx].astype(np.int64))
        
        return img, target


# === Data Augmentation ===
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Polar images: rotation augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

val_transform = None

# === Load Data ===
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# === Vision Transformer Model ===
try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    print("\n✓ Using torchvision ViT_B_16 with pretrained weights")
except ImportError:
    print("\n⚠ torchvision ViT not available, using timm instead")
    import timm
    vit_model_fn = lambda: timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)


class ViTMultiHead(nn.Module):
    """
    Vision Transformer with multi-head classification for antenna architecture prediction.
    Each ring (5 total) has its own classification head (11 classes: 0-10 elements).
    """
    def __init__(self, num_rings=5, num_classes_per_ring=11, pretrained=True):
        super(ViTMultiHead, self).__init__()
        self.num_rings = num_rings
        self.num_classes_per_ring = num_classes_per_ring
        
        # Load pretrained ViT backbone
        try:
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            vit = vit_b_16(weights=weights)
        except:
            # Fallback to timm
            import timm
            vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        
        # Remove classification head, keep feature extractor
        if hasattr(vit, 'heads'):
            # torchvision ViT
            self.backbone = nn.Sequential(*list(vit.children())[:-1])
            hidden_dim = vit.heads[0].in_features
        else:
            # timm ViT
            self.backbone = vit
            hidden_dim = vit.embed_dim if hasattr(vit, 'embed_dim') else 768
        
        # Multi-head classification: one head per ring
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes_per_ring)
            for _ in range(num_rings)
        ])
        
        print(f"✓ ViT backbone initialized with hidden_dim={hidden_dim}")
        print(f"✓ Multi-head classifier: {num_rings} heads × {num_classes_per_ring} classes")
    
    def forward(self, x):
        """
        x: (B, 3, 240, 240)
        returns: list of (B, 11) logits for each ring
        """
        # Backbone forward pass
        features = self.backbone(x)  # (B, hidden_dim)
        
        # Multi-head classification
        outputs = [head(features) for head in self.heads]  # list of (B, 11)
        
        return outputs


# === Model, Loss, Optimizer ===
model = ViTMultiHead(num_rings=5, num_classes_per_ring=11, pretrained=True).to(DEVICE)
print(f"\nModel moved to {DEVICE}")

# Loss: CrossEntropy for each ring independently
criterion = nn.CrossEntropyLoss()

# Optimizer: AdamW with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler: warmup then cosine annealing
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

# === Training Loop ===
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = os.path.join(script_dir, 'vit_best_model.pth')
checkpoint_path = os.path.join(script_dir, 'vit_last_checkpoint.pth')

print("\n" + "="*80)
print("Starting training...")
print("="*80 + "\n")

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # === Training Phase ===
    model.train()
    train_loss = 0.0
    train_accuracy = [0.0] * 5  # per-ring accuracy
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(DEVICE)  # (B, 3, 240, 240)
        targets = targets.to(DEVICE)  # (B, 5)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # list of (B, 11)
        
        # Compute loss: sum of CE losses across all rings
        loss = 0.0
        for ring_idx in range(5):
            loss += criterion(outputs[ring_idx], targets[:, ring_idx])
        loss /= 5  # Average across rings
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        # Compute per-ring accuracy
        for ring_idx in range(5):
            pred = outputs[ring_idx].argmax(dim=1)
            train_accuracy[ring_idx] += (pred == targets[:, ring_idx]).float().mean().item()
        
        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    train_loss /= len(train_loader)
    train_accuracy = [acc / len(train_loader) for acc in train_accuracy]
    train_losses.append(train_loss)
    
    # === Validation Phase ===
    model.eval()
    val_loss = 0.0
    val_accuracy = [0.0] * 5
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
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
    
    # === Logging ===
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Train Acc: {np.mean(train_accuracy):.4f}, Val Acc: {np.mean(val_accuracy):.4f}")
    print(f"  Per-ring Val Acc: {[f'{acc:.3f}' for acc in val_accuracy]}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Update scheduler
    scheduler.step()
    
    # === Checkpointing ===
    # Save last checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    
    print()

end_time = time.time()
print("="*80)
print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {best_model_path}")
print("="*80)

# === Plot Loss Curves ===
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
