import torch
from hubert.customtokenizer import auto_train
import os
from config import datasets_path

if __name__ == "__main__":
    print("start train")
    auto_train(
        datasets_path,
        load_model=os.path.join(datasets_path, "model.pth"),
        save_epochs=20,
    )
