from bark.generation import load_codec_model, _grab_best_device
from encodec.utils import convert_audio

import torchaudio
import torch
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from config import tokenizer_path, hubert_model_path

device = torch.cuda.is_available() and "cuda" or "cpu"

model = load_codec_model(use_gpu=True)
hubert_model = CustomHubert(checkpoint_path=hubert_model_path).to(device)

tokenizer = CustomTokenizer.load_from_checkpoint(
    tokenizer_path, map_location=device
).to(device)
audio_file_path = ""
# Load and pre-process the audio waveform
wav, sr = torchaudio.load(audio_file_path)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.to(device)

semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat(
    [encoded[0] for encoded in encoded_frames], dim=-1
).squeeze()  # [n_q, T]

# move codes to cpu
codes = codes.cpu().numpy()
# move semantic tokens to cpu
semantic_tokens = semantic_tokens.cpu().numpy()

import numpy as np

voice_name = "cc_new"
output_path = "bark/assets/prompts/" + voice_name + ".npz"
np.savez(
    output_path,
    fine_prompt=codes,
    coarse_prompt=codes[:2, :],
    semantic_prompt=semantic_tokens,
)
