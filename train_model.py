"""Entry point: load data, train/compare the 3 models, save the best one to model/.

Run from the project root:
    python train_model.py
"""

from utils.train import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()
