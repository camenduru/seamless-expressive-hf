#!/usr/bin/env python

import os
import pathlib
import tempfile

import gradio as gr
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from fairseq2.data import Collater, SequenceData
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)
from fairseq2.generation import SequenceGeneratorOptions
from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType, Device
from huggingface_hub import snapshot_download
from seamless_communication.inference import BatchedSpeechOutput, Translator
from seamless_communication.models.generator.loader import load_pretssel_vocoder_model
from seamless_communication.models.unity import (
    UnitTokenizer,
    load_gcmvn_stats,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from torch.nn import Module

DESCRIPTION = """\
# Seamless Expressive


"""

CACHE_EXAMPLES = os.getenv("CACHE_EXAMPLES") == "1" and torch.cuda.is_available()

CHECKPOINTS_PATH = pathlib.Path(os.getenv("CHECKPOINTS_PATH", "/home/user/app/models"))
if not CHECKPOINTS_PATH.exists():
    snapshot_download(repo_id="meta-private/SeamlessExpressive", repo_type="model", local_dir=CHECKPOINTS_PATH)

# Ensure that we do not have any other environment resolvers and always return
# "demo" for demo purposes.
asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")

# Construct an `InProcAssetMetadataProvider` with environment-specific metadata
# that just overrides the regular metadata for "demo" environment. Note the "@demo" suffix.
demo_metadata = [
    {
        "name": "seamless_expressivity@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/m2m_expressive_unity.pt",
        "char_tokenizer": f"file://{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
    {
        "name": "vocoder_pretssel@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/pretssel_melhifigan_wm-final.pt",
    },
]

asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))
# Now when we retrieve the card, the corresponding fields should come from the
# demo metadata.
# card = asset_store.retrieve_card("vocoder_pretssel")

# print(card.field("checkpoint").as_uri())


class PretsselGenerator(Module):
    def __init__(
        self,
        pretssel_name_or_card: str,
        unit_tokenizer: UnitTokenizer,
        device: Device,
        dtype: DataType = torch.float16,
    ):
        super().__init__()
        # Load the model.
        if device == torch.device("cpu"):
            dtype = torch.float32

        self.device = device
        self.dtype = dtype

        self.pretssel_model = load_pretssel_vocoder_model(
            pretssel_name_or_card,
            device=device,
            dtype=dtype,
        )
        self.pretssel_model.eval()

        vocoder_model_card = asset_store.retrieve_card(pretssel_name_or_card)
        self.output_sample_rate = vocoder_model_card.field("sample_rate").as_(int)

        self.unit_tokenizer = unit_tokenizer
        self.unit_collate = Collater(pad_value=unit_tokenizer.vocab_info.pad_idx)
        self.duration_collate = Collater(pad_value=0)

    @torch.inference_mode()
    def predict(
        self,
        units: list[list[int]],
        tgt_lang: str,
        prosody_encoder_input: SequenceData,
    ) -> BatchedSpeechOutput:
        audio_wavs = []
        unit_eos_token = torch.tensor(
            [self.unit_tokenizer.vocab_info.eos_idx],
            device=self.device,
        )

        prosody_input_seqs = prosody_encoder_input["seqs"]
        prosody_input_lens = prosody_encoder_input["seq_lens"]

        for i, u in enumerate(units):
            unit = torch.tensor(u).to(unit_eos_token)

            # adjust the control symbols for the embedding
            unit += 4
            unit = torch.cat([unit, unit_eos_token], dim=0)

            unit, duration = torch.unique_consecutive(unit, return_counts=True)

            # adjust for the last eos token
            duration[-1] = 0

            duration *= 2

            prosody_input_seq = prosody_input_seqs[i][: prosody_input_lens[i]]

            audio_wav = self.pretssel_model(
                unit,
                tgt_lang,
                prosody_input_seq,
                durations=duration.unsqueeze(0),
            )

            audio_wavs.append(audio_wav)

        return BatchedSpeechOutput(
            units=units,
            audio_wavs=audio_wavs,
            sample_rate=self.output_sample_rate,
        )


LANGUAGE_CODE_TO_NAME = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}
LANGUAGE_NAME_TO_CODE = {v: k for k, v in LANGUAGE_CODE_TO_NAME.items()}


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


MODEL_NAME = "seamless_expressivity"
VOCODER_NAME = "vocoder_pretssel"

text_tokenizer = load_unity_text_tokenizer(MODEL_NAME)
unit_tokenizer = load_unity_unit_tokenizer(MODEL_NAME)

_gcmvn_mean, _gcmvn_std = load_gcmvn_stats(VOCODER_NAME)
gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

translator = Translator(
    MODEL_NAME,
    vocoder_name_or_card=None,
    device=device,
    text_tokenizer=text_tokenizer,
    dtype=dtype,
)

text_generation_opts, unit_generation_opts = SequenceGeneratorOptions(
    beam_size=5, soft_max_seq_len=None
), SequenceGeneratorOptions(beam_size=5, soft_max_seq_len=(25, 50))

pretssel_generator = PretsselGenerator(
    VOCODER_NAME,
    unit_tokenizer=unit_tokenizer,
    device=device,
    dtype=dtype,
)

decode_audio = AudioDecoder(dtype=torch.float32, device=device)

convert_to_fbank = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=False,
    device=device,
    dtype=dtype,
)


def normalize_fbank(data: WaveformToFbankOutput) -> WaveformToFbankOutput:
    fbank = data["fbank"]
    std, mean = torch.std_mean(fbank, dim=0)
    data["fbank"] = fbank.subtract(mean).divide(std)
    data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
    return data


collate = Collater(pad_value=0, pad_to_multiple=1)


AUDIO_SAMPLE_RATE = 44100
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds


def preprocess_audio(input_audio_path: str) -> None:
    arr, org_sr = torchaudio.load(input_audio_path)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio_path, new_arr, sample_rate=AUDIO_SAMPLE_RATE)


def run(input_audio_path: str, target_language: str) -> tuple[str, str]:
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    preprocess_audio(input_audio_path)

    with pathlib.Path(input_audio_path).open("rb") as fb:
        block = MemoryBlock(fb.read())
        example = decode_audio(block)

    example = convert_to_fbank(example)
    example = normalize_fbank(example)
    example = collate(example)

    prosody_encoder_input = example["gcmvn_fbank"]
    text_output, unit_output = translator.predict(
        example["fbank"],
        "S2ST",
        target_language_code,
        src_lang=None,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=unit_generation_opts,
        unit_generation_ngram_filtering=False,
        duration_factor=1.0,
        prosody_encoder_input=prosody_encoder_input,
    )
    speech_output = pretssel_generator.predict(
        unit_output.units,
        tgt_lang=target_language_code,
        prosody_encoder_input=prosody_encoder_input,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(
            f.name,
            speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
            sample_rate=speech_output.sample_rate,
        )
    return f.name, str(text_output[0])


TARGET_LANGUAGE_NAMES = [
    "English",
    "French",
    "German",
    "Italian",
    "Mandarin Chinese",
    "Spanish",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech", type="filepath")
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=TARGET_LANGUAGE_NAMES,
                    value="French",
                )
            btn = gr.Button()
        with gr.Column():
            with gr.Group():
                output_audio = gr.Audio(label="Translated speech")
                output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[
            ["assets/sample_input.mp3", "French"],
            ["assets/sample_input.mp3", "Mandarin Chinese"],
            ["assets/sample_input_2.mp3", "French"],
            ["assets/sample_input_2.mp3", "Spanish"],
        ],
        inputs=[input_audio, target_language],
        outputs=[output_audio, output_text],
        fn=run,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    btn.click(
        fn=run,
        inputs=[input_audio, target_language],
        outputs=[output_audio, output_text],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch()
