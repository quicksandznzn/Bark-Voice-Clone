import os

now_dir = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(now_dir, "datasets")
datasets_file_path = os.path.join(datasets_path, "part_1.txt")
semantic_path = os.path.join(datasets_path, "semantic")
wav_path = os.path.join(datasets_path, "wav")
prepared_path = os.path.join(datasets_path, "prepared")
ready_path = os.path.join(datasets_path, "ready")

bark_model_path = os.path.join(now_dir, "models")
hubert_model_path = os.path.join(now_dir, "models/hubert.pt")
tokenizer_path = os.path.join(now_dir, "models/zh_tokenizer.pth")
