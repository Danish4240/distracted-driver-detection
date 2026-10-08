import os
import torch

class Config:
    SEED = 42
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
    DRIVER_CSV = os.path.join(RAW_DATA_DIR, "driver_imgs_list.csv")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    
    NUM_CLASSES = 10
    CLASS_LABELS = {
        "c0": "Safe Driving",
        "c1": "Texting - Right",
        "c2": "Talking Phone - Right",
        "c3": "Texting - Left",
        "c4": "Talking Phone - Left",
        "c5": "Operating Radio",
        "c6": "Drinking",
        "c7": "Reaching Behind",
        "c8": "Hair and Makeup",
        "c9": "Talking to Passenger"
    }

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Must be 0 on Windows to prevent silent worker crashes
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"