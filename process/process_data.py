import random
import gdown
from zipfile import ZipFile
from config import datasets_path
import os
import json

allowed_chars = "0123456789!@#$%^&*()-_+=\"':;[]{}/<>,.`~\n\\"


def download_wiki_datasets(file_path, output_folder):
    if not os.path.isfile(file_path):
        zip_file_path = os.path.join(output_folder, "downloaded_file.zip")
        gdown.download(id=file_path, output=zip_file_path, quiet=False)
    else:
        zip_file_path = file_path
    # Extract the contents of the ZIP file
    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    # Delete the ZIP file
    datasets_file_path = os.path.join(output_folder, "datasets.txt")
    read_json_lines(output_folder, datasets_file_path)
    dataset_part_folder = os.path.join(output_folder, "part")
    if not os.path.isdir(dataset_part_folder):
        os.mkdir(dataset_part_folder)
    split_large_file(datasets_file_path, dataset_part_folder, chunk_size=50)
    txt_files = [
        f
        for f in os.listdir(dataset_part_folder)
        if f.endswith(".txt") and os.path.isfile(os.path.join(dataset_part_folder, f))
    ]
    return txt_files


def read_book(book):
    with open(book, "r", encoding="utf-8") as f:
        return f.read()


def is_cjk(character):
    return any(
        [
            start <= ord(character) <= end
            for start, end in [
                (4352, 4607),
                (11904, 42191),
                (43072, 43135),
                (44032, 55215),
                (63744, 64255),
                (65072, 65103),
                (65381, 65500),
                (131072, 196607),
            ]
        ]
    )


def filter_data(data):
    print("Filtering data")
    return "".join([char for char in data if char in allowed_chars or is_cjk(char)])


def load_book(book):
    print(f"Loading book into ram")
    return filter_data(str(read_book(book)))


def random_split_chunk(data, size=14, spliter=None):
    if spliter is not None:
        data = data.split(spliter)
    index = random.randrange(0, len(data))

    if spliter is not None:
        return spliter.join(data[index : index + size])
    else:
        return data[index : index + size]


def write_text_to_file(text, output_file):
    with open(output_file, "a", encoding="utf-8") as file:
        if text.strip():
            file.write(text + "\n")


def read_json_lines(folder_path, output_file):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if "wiki_" in file_name:
                file_path = os.path.join(root, file_name)
                print(f"Reading file: {file_path}")

                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        try:
                            json_content = json.loads(line)
                            if "text" in json_content:
                                text_content = json_content["text"]
                                write_text_to_file(text_content, output_file)
                        except json.JSONDecodeError as e:
                            print(
                                f"Error decoding JSON in {file_path}, line: {line}\nError: {e}"
                            )


def split_large_file(input_file, output_folder, chunk_size=50):
    chunk_size_mb = chunk_size * 1024 * 1024
    with open(input_file, "rb") as f:
        part_num = 1
        while True:
            chunk = f.read(chunk_size_mb)
            if not chunk:
                break

            output_file = os.path.join(output_folder, f"part_{part_num}.txt")
            with open(output_file, "w", encoding="utf-8") as part_file:
                part_file.write(chunk.decode("utf-8", errors=".gitignore"))

            part_num += 1


if __name__ == "__main__":
    txt_files = download_wiki_datasets(
        "/Users/quicksandzn/Downloads/wiki_zh_2019.zip", datasets_path
    )
