# import torch
# import torchaudio
# from fairseq2.assets import InProcAssetMetadataProvider, asset_store
# from fairseq2.data import Collater, SequenceData
# from fairseq2.data.audio import (
#     AudioDecoder,
#     WaveformToFbankConverter,
#     WaveformToFbankOutput,
# )
# from fairseq2.generation import SequenceGeneratorOptions
# from fairseq2.memory import MemoryBlock
# from fairseq2.typing import DataType, Device
# from huggingface_hub import snapshot_download
# from seamless_communication.inference import BatchedSpeechOutput, Translator
# from seamless_communication.models.generator.loader import load_pretssel_vocoder_model
# from seamless_communication.models.unity import (
#     UnitTokenizer,
#     load_gcmvn_stats,
#     load_unity_text_tokenizer,
#     load_unity_unit_tokenizer,
# )
# from torch.nn import Module

# class PretsselGenerator(Module):
#     def __init__(
#         self,
#         pretssel_name_or_card: str,
#         unit_tokenizer: UnitTokenizer,
#         device: Device,
#         dtype: DataType = torch.float16,
#     ):
#         super().__init__()
#         # Load the model.
#         if device == torch.device("cpu"):
#             dtype = torch.float32


#         self.device = device
#         self.dtype = dtype

#         self.pretssel_model = load_pretssel_vocoder_model(
#             pretssel_name_or_card,
#             device=device,
#             dtype=dtype,
#         )
#         self.pretssel_model.eval()

#         vocoder_model_card = asset_store.retrieve_card(pretssel_name_or_card)
#         self.output_sample_rate = vocoder_model_card.field("sample_rate").as_(int)

#         self.unit_tokenizer = unit_tokenizer
#         self.unit_collate = Collater(pad_value=unit_tokenizer.vocab_info.pad_idx)
#         self.duration_collate = Collater(pad_value=0)

#     @torch.inference_mode()
#     def predict(
#         self,
#         units: list[list[int]],
#         tgt_lang: str,
#         prosody_encoder_input: SequenceData,
#     ) -> BatchedSpeechOutput:
#         audio_wavs = []
#         unit_eos_token = torch.tensor(
#             [self.unit_tokenizer.vocab_info.eos_idx],
#             device=self.device,
#         )

#         prosody_input_seqs = prosody_encoder_input["seqs"]
#         prosody_input_lens = prosody_encoder_input["seq_lens"]

#         for i, u in enumerate(units):
#             unit = torch.tensor(u).to(unit_eos_token)

#             # adjust the control symbols for the embedding
#             unit += 4
#             unit = torch.cat([unit, unit_eos_token], dim=0)

#             unit, duration = torch.unique_consecutive(unit, return_counts=True)

#             # adjust for the last eos token
#             duration[-1] = 0

#             duration *= 2

#             prosody_input_seq = prosody_input_seqs[i][: prosody_input_lens[i]]

#             audio_wav = self.pretssel_model(
#                 unit,
#                 tgt_lang,
#                 prosody_input_seq,
#                 durations=duration.unsqueeze(0),
#             )

#             audio_wavs.append(audio_wav)

#         return BatchedSpeechOutput(
#             units=units,
#             audio_wavs=audio_wavs,
#             sample_rate=self.output_sample_rate,
#         )


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
