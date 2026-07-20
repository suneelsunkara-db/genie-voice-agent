"""Language-code maps for the realtime API's 24 languages across benchmark datasets.

The realtime API reports 24 BCP-47 primary subtags. Each benchmark uses its own
locale convention, so we map our canonical 2-letter codes to:
  - FLEURS  config names (e.g. ``en_us``)          -> google/fleurs
  - FLORES-200 codes (e.g. ``eng_Latn``)           -> facebook/2M-Belebele
  - CCFQA   ISO-3 codes (e.g. ``eng``)             -> yxdu/ccfqa
"""
from __future__ import annotations

# Canonical set reported by GET /v1/languages.
SUPPORTED = [
    "ar", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "hi",
    "id", "it", "ja", "ko", "ms", "pl", "pt", "ru", "es", "sv",
    "fil", "th", "tr", "vi",
]

# 2-letter -> FLEURS config (google/fleurs).
FLEURS = {
    "ar": "ar_eg",
    "zh": "cmn_hans_cn",
    "da": "da_dk",
    "nl": "nl_nl",
    "en": "en_us",
    "fi": "fi_fi",
    "fr": "fr_fr",
    "de": "de_de",
    "el": "el_gr",
    "hi": "hi_in",
    "id": "id_id",
    "it": "it_it",
    "ja": "ja_jp",
    "ko": "ko_kr",
    "ms": "ms_my",
    "pl": "pl_pl",
    "pt": "pt_br",
    "ru": "ru_ru",
    "es": "es_419",
    "sv": "sv_se",
    "fil": "fil_ph",
    "th": "th_th",
    "tr": "tr_tr",
    "vi": "vi_vn",
}

# 2-letter -> FLORES-200 code (facebook/2M-Belebele, facebook/belebele).
# Note: 2M-Belebele has no Modern Standard Arabic (arb_Arab) or Malay
# (zsm_Latn), so "ar" and "ms" are intentionally omitted — substituting a
# related language (Egyptian Arabic / Indonesian) would report misleading
# scores. FLEURS and CCFQA still cover those languages where available.
BELEBELE = {
    "zh": "zho_Hans",
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "en": "eng_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "sv": "swe_Latn",
    "fil": "tgl_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
}

# 2-letter -> CCFQA ISO-3 code (yxdu/ccfqa). Only 7 of our languages exist there.
CCFQA = {
    "zh": "cmn",
    "en": "eng",
    "fr": "fra",
    "ja": "jpn",
    "ko": "kor",
    "ru": "rus",
    "es": "spa",
}

# Scripts written without word spacing — WER on whitespace tokens is meaningless,
# so CER is the primary metric for these.
SPACELESS_SCRIPTS = {"zh", "ja", "th"}


def dataset_languages(dataset: str) -> dict[str, str]:
    table = {"fleurs": FLEURS, "belebele": BELEBELE, "ccfqa": CCFQA}
    if dataset not in table:
        raise KeyError(f"unknown dataset: {dataset}")
    return table[dataset]


def resolve_languages(dataset: str, requested: list[str] | None) -> list[str]:
    """Filter requested 2-letter codes to those the dataset actually covers."""
    available = dataset_languages(dataset)
    if not requested:
        return [code for code in SUPPORTED if code in available]
    resolved = [code for code in requested if code in available]
    return resolved
