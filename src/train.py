import sys
import os

# Ensure src root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.mlops.trainer import train_model

if __name__ == "__main__":
    print("=========================================")
    print(" Starting ML Model Training & Pipeline  ")
    print("=========================================")
    train_model(data_path="data/dataset.csv")
