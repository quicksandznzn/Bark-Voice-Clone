from bark.api import generate_audio
from bark.generation import (
    SAMPLE_RATE,
    preload_models,
    codec_decode,
    generate_text_semantic,
    generate_fine,
    generate_coarse,
)
from scipy.io.wavfile import write as write_wav


class VoiceClone:
    def __init__(self, model_path):
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=False,
            path=model_path,
        )

    def simple_generate(self, text, voice_name):
        return generate_audio(text, history_prompt=voice_name)

    def generate(self, text, voice_name):
        x_semantic = generate_text_semantic(
            text,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )

        x_coarse_gen = generate_coarse(
            x_semantic,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            history_prompt=voice_name,
            temp=0.5,
        )
        return codec_decode(x_fine_gen)

    def save(self, audio_data, file_path):
        write_wav(file_path, SAMPLE_RATE, audio_data)


if __name__ == "__main__":
    from config import bark_model_path

    text = "大家好，欢迎你来到我的地盘"
    voice_name = "cc_new"
    vc = VoiceClone(model_path=bark_model_path)
    audio_data = vc.generate(text, voice_name)
    vc.save(audio_data, "audio.wav")
