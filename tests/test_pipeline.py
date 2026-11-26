import json
import math
import re
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

ABBREV_RE = re.compile(
    r"\b(?:o\.a|bijv|d\.w\.z|i\.p\.v|m\.b\.v|m\.a\.w|t\.o\.v|t\.a\.v|"
    r"i\.v\.m|n\.a\.v|o\.b\.v|o\.i\.d|e\.d|enz|etc|nl|dr|drs|prof|"
    r"mr|ir|dhr|mw|mevr|fam|blz|pag|nr|ca|st|max|min|tel|a\.u\.b|s\.v\.p)\.",
    re.IGNORECASE,
)
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)|\r?\n+")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")


def load_assets():
    with open(WEB_DIR / "dictionary.json", "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    with open(WEB_DIR / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    session = ort.InferenceSession(str(WEB_DIR / "leesniveau_model.onnx"))
    return dictionary, meta, session


def resolve_entry(dictionary, word):
    lower = word.lower().replace("’", "'")
    for candidate in (
        lower,
        lower[:-2] if lower.endswith("'s") and len(lower) > 3 else None,
        lower.replace("-", "") if "-" in lower else None,
    ):
        if not candidate or candidate not in dictionary:
            continue
        entry = dictionary[candidate]
        if "l" in entry:
            base = dictionary.get(entry["l"])
            if base:
                c = base["c"] if "c" in base else entry.get("c")
                return (entry["l"], base.get("z", 0.0), c)
        return (candidate, entry.get("z", 0.0), entry.get("c"))
    return None


def extract_features(dictionary, meta, text):
    cleaned = ABBREV_RE.sub(lambda m: m.group(0).replace(".", "·"), text.strip().replace("’", "'"))
    sents = [w for s in SENT_SPLIT_RE.split(cleaned) if s.strip() and (w := WORD_RE.findall(s))]
    words = [w for s in sents for w in s]
    n_words, n_sents = len(words), len(sents)

    avg_sent_len = n_words / n_sents
    max_sent_len = float(max(len(s) for s in sents))
    std_sent_len = float(np.std([len(s) for s in sents]))
    avg_word_len = sum(len(w) for w in words) / n_words
    long_word_ratio = sum(1 for w in words if len(w) >= 10) / n_words

    lemmas, zipfs, nt2_scores, oov = set(), [], [], 0
    for w in words:
        res = resolve_entry(dictionary, w)
        if res:
            lemma, z, c = res
            lemmas.add(lemma)
            if z > 0:
                zipfs.append(z)
            if c is not None:
                nt2_scores.append(c)
            else:
                oov += 1
        else:
            lemmas.add(w.lower())
            oov += 1

    guiraud_ttr = len(lemmas) / math.sqrt(n_words)
    p10_zipf = float(np.percentile(zipfs, 10)) if zipfs else 0.0
    rare_ratio = sum(1 for z in zipfs if z < meta["rare_zipf_threshold"]) / len(zipfs) if zipfs else 0.0
    nt2_mean = sum(nt2_scores) / len(nt2_scores) if nt2_scores else 4.0

    return np.array(
        [[
            avg_sent_len, max_sent_len, std_sent_len,
            avg_word_len, long_word_ratio, guiraud_ttr,
            p10_zipf, rare_ratio, nt2_mean, oov / n_words,
        ]],
        dtype=np.float32,
    )


def test_dictionary_invariants():
    dictionary, _, _ = load_assets()
    assert len(dictionary) > 120_000
    for word, entry in dictionary.items():
        if "l" in entry:
            target = dictionary.get(entry["l"])
            assert target is not None and "d" in target and "z" in target, f"Broken pointer: {word}"
        elif "d" in entry:
            assert isinstance(entry["z"], (int, float))
        else:
            assert entry.get("c") in {0, 1, 2, 3, 4}, f"Invalid NT2-only entry: {word}"


def test_homograph_levels():
    dictionary, _, _ = load_assets()
    for word in ("meer", "was", "kan", "wil"):
        res = resolve_entry(dictionary, word)
        assert res is not None and res[2] == 0, f"Expected A1 (0) for {word}, got {res}"


def test_model_meta_and_onnx_inference():
    dictionary, meta, session = load_assets()
    assert len(meta["features"]) == 10
    assert meta["thresholds"] == sorted(meta["thresholds"])
    assert len(meta["thresholds"]) == 5

    nos_text = (
        "Het NOS Journaal in Makkelijke Taal heeft minder onderwerpen dan de andere journaals. "
        "De presentator geeft meer uitleg en er worden geen moeilijke woorden gebruikt. "
        "Ook het tempo en het gebruik van beeld is rustiger. "
        "Kijkers van het eerdere weekjournaal (NOS Nieuws van de Week) vinden dat prettig, zo blijkt uit hun reacties."
    )
    feats = extract_features(dictionary, meta, nos_text)
    input_name = session.get_inputs()[0].name
    raw_score = float(session.run(None, {input_name: feats})[0].item())
    pred_idx = int(np.digitize([raw_score], meta["thresholds"])[0])

    assert meta["labels"][pred_idx] == "A2"
    assert 0.8 <= raw_score <= 1.5
