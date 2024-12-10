import os

def read_corpus(folder_path: str, type: str="train"):
    # type : "train", "dev" or "test"
    en_file_path = os.path.join(folder_path, type, f"{type}.en")
    vi_file_path = os.path.join(folder_path, type, f"{type}.vi")
    with open(en_file_path, "r") as f:
        en_dataset = f.readlines()
        en_dataset = [line.strip() for line in en_dataset]
    with open(vi_file_path, "r") as f:
        vi_dataset = f.readlines()
        vi_dataset = [line.strip() for line in vi_dataset]
    return en_dataset, vi_dataset