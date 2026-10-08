import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.dataset import get_driver_splits, create_dataloaders
from src.models import build_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training Batch", leave=True, file=sys.stdout):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation Batch", leave=True, file=sys.stdout):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
    
    print(f"Loading driver splits from {Config.DRIVER_CSV}...")
    df = get_driver_splits(Config.DRIVER_CSV)

    train_loader, val_loader = create_dataloaders(df, val_fold=0)

    print(f"Initializing model on device: {Config.DEVICE}")
    model = build_model(model_name="efficientnet_b0", pretrained=True)

    weights_path = os.path.join(Config.WEIGHTS_DIR, "best_model.pth")
    start_epoch = 1
    best_val_loss = float("inf")

    # Load existing checkpoint if available to resume training
    if os.path.exists(weights_path):
        print(f"--> Found existing checkpoint at {weights_path}. Loading weights to resume...")
        model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
        
        # Resume training starting from Epoch 3
        start_epoch = 3
        print(f"--> Resuming training starting from Epoch {start_epoch}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    total_epochs = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Fast-forward scheduler to match resumed epoch state
    for _ in range(start_epoch - 1):
        scheduler.step()

    print("\n--- Resuming Model Training ---")
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} ---")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        scheduler.step()

        print(
            f"Epoch [{epoch}/{total_epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            print(f"  --> Saved new best checkpoint to {weights_path}")


if __name__ == "__main__":
    main()