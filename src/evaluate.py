import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.dataset import get_driver_splits, create_dataloaders
from src.models import build_model


def evaluate_model():
    print("Loading validation dataset...")
    df = get_driver_splits(Config.DRIVER_CSV)
    _, val_loader = create_dataloaders(df, val_fold=0)

    weights_path = os.path.join(Config.WEIGHTS_DIR, "best_model.pth")
    print(f"Loading weights from {weights_path}...")
    
    model = build_model(model_name="efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", file=sys.stdout):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = [Config.CLASS_LABELS[f"c{i}"] for i in range(10)]

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Distracted Driver Detection - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_fig_path = os.path.join(Config.BASE_DIR, "confusion_matrix.png")
    plt.savefig(save_fig_path)
    print(f"Saved confusion matrix plot to {save_fig_path}")


if __name__ == "__main__":
    evaluate_model()