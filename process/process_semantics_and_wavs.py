import os.path
import uuid
from bark import text_to_semantic
from hubert.pre_kmeans_hubert import CustomHubert
from process_data import load_book, random_split_chunk
import os
import random
import concurrent.futures
import numpy
from scipy.io import wavfile
import shutil
from bark.generation import preload_models, SAMPLE_RATE
from bark.api import semantic_to_waveform
from config import (
    hubert_model_path,
    bark_model_path,
    semantic_path,
    wav_path,
    datasets_file_path,
    prepared_path,
    ready_path,
)
import torchaudio
import torch

device = torch.cuda.is_available() and "cuda" or "cpu"

preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
    path=bark_model_path,
)

if not os.path.isdir(semantic_path):
    os.mkdir(semantic_path)

if not os.path.isdir(wav_path):
    os.mkdir(wav_path)

if not os.path.isdir(prepared_path):
    os.mkdir(prepared_path)

if not os.path.isdir(ready_path):
    os.mkdir(ready_path)


def process_semantic():
    loaded_data = load_book(datasets_file_path)
    while 1:
        filename = uuid.uuid4().hex + ".npy"
        file_name = os.path.join(semantic_path, filename)
        text = ""
        while not len(text) > 0:
            text = random_split_chunk(loaded_data)  # Obtain a short chunk of text
            text = text.strip()
        print("Generating semantics for text:", text)
        semantics = text_to_semantic(
            text, temp=round(random.uniform(0.6, 0.8), ndigits=2)
        )
        numpy.save(file_name, semantics)


def process_wav(f):
    real_name = ".".join(f.split(".")[:-1])  # Cut off the extension
    file_name = os.path.join(semantic_path, f)
    out_file = os.path.join(wav_path, f"{real_name}.wav")

    # Don't process files that have already been processed
    if not os.path.isfile(out_file) and os.path.isfile(file_name):
        print(f"Processing {f}")
        wav = semantic_to_waveform(
            numpy.load(file_name), temp=round(random.uniform(0.6, 0.8), ndigits=2)
        )
        wavfile.write(out_file, SAMPLE_RATE, wav)


def process_wav_with_thread(max_workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        file_list = os.listdir(semantic_path)
        executor.map(process_wav, file_list)


def prepared():
    new_offset = 0
    for wave_file in os.listdir(wav_path):
        filename = ".".join(wave_file.strip().split(".")[:-1])
        semantic_file = os.path.join(semantic_path, f"{filename}.npy")
        if os.path.isfile(semantic_file):
            print(f"Process {filename}")
            shutil.copyfile(
                os.path.join(wav_path, f"{filename}.wav"),
                os.path.join(prepared_path, f"{new_offset}_wav.wav"),
            )
            shutil.copyfile(
                os.path.join(semantic_path, f"{filename}.npy"),
                os.path.join(prepared_path, f"{new_offset}_semantic.npy"),
            )

            new_offset += 1


def ready():
    hubert_model = CustomHubert(checkpoint_path=hubert_model_path, device=device)
    wav_string = "_wav.wav"
    sem_string = "_semantic.npy"

    for input_file in os.listdir(prepared_path):
        input_path = os.path.join(prepared_path, input_file)
        if input_file.endswith(wav_string):
            file_num = int(input_file[: -len(wav_string)])
            fname = f"{file_num}_semantic_features.npy"
            output_path = os.path.join(ready_path, fname)
            print("Processing", input_file)
            if os.path.isfile(output_path):
                continue
            wav, sr = torchaudio.load(input_path)

            if wav.shape[0] == 2:  # Stereo to mono if needed
                wav = wav.mean(0, keepdim=True)

            output = hubert_model.forward(wav, input_sample_hz=sr)
            out_array = output.cpu().numpy()
            numpy.save(output_path, out_array)
        elif input_file.endswith(sem_string):
            output_path = os.path.join(ready_path, input_file)
            if os.path.isfile(output_path):
                continue
            shutil.copy(input_path, output_path)
    print("All set! We're ready to train!")


if __name__ == "__main__":
    process_semantic()
    process_wav_with_thread()
    prepared()
    ready()
