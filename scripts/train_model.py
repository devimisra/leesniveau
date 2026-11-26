import json
import math
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATA_PATH = "../data/local_elg_cefr_nl.json"
DICT_PATH = "../web/dictionary.json"
ONNX_OUT = "../web/leesniveau_model.onnx"
META_OUT = "../web/model_meta.json"

MODEL_VERSION = "1.5.1"
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
NT2_LEVELS = ["A1", "A2", "B1", "B2", "C1"]
CEFR_MAP = {lvl: i for i, lvl in enumerate(CEFR_ORDER)}

FEATURES = [
    "avg_sent_len", "max_sent_len", "std_sent_len",
    "avg_word_len", "long_word_ratio", "guiraud_ttr",
    "p10_zipf", "rare_word_ratio",
    "nt2_mean_first", "rare_nt2_oov_ratio", "common_nt2_oov_ratio",
]

ABBREV_RE = re.compile(
    r"\b(?:o\.a|bijv|d\.w\.z|i\.p\.v|m\.b\.v|m\.a\.w|t\.o\.v|t\.a\.v|"
    r"i\.v\.m|n\.a\.v|o\.b\.v|o\.i\.d|e\.d|enz|etc|nl|dr|drs|prof|"
    r"mr|ir|dhr|mw|mevr|fam|blz|pag|nr|ca|st|max|min|tel|a\.u\.b|s\.v\.p)\.",
    re.IGNORECASE,
)
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)|\r?\n+")
CLAUSE_SPLIT_RE = re.compile(
    r"[,;:()–—]+|\b(?:en|maar|want|of|dus|omdat|dat|toen|terwijl|hoewel|"
    r"indien|zodat|waardoor|waarbij|als|wanneer|omtrent)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
RARE_ZIPF_THRESHOLD = 4.0

SEPARABLE_PARTICLES = {
    "aan", "achter", "af", "bij", "binnen", "boven", "buiten", "door", "gelijk", "goed",
    "in", "langs", "mee", "mis", "na", "neer", "om", "onder", "op", "open", "over",
    "rond", "samen", "schoon", "stil", "tegen", "terug", "thuis", "toe", "tussen",
    "uit", "vast", "voor", "voort", "vrij", "weg", "weer",
}

AUX_OR_STOP_VERBS = {
    "zijn", "hebben", "worden", "kunnen", "moeten", "zullen", "willen", "mogen",
    "de", "het", "een", "dat", "die", "wat", "wie", "waar", "hoe",
}

VERB_INF_RE = re.compile(r"(?:en|aan|oen|ijn|ien)$")


def is_verb_infinitive(lemma: str) -> bool:
    return bool(VERB_INF_RE.search(lemma)) and len(lemma) >= 5


print("1. Loading unified dictionary.json and dataset...")
with open(DICT_PATH, "r", encoding="utf-8") as f:
    dictionary = json.load(f)

df = pd.read_json(DATA_PATH)
df = df.dropna(subset=["cefr_level", "text"])
if "category" in df.columns:
    df = df[df["category"] == "reference"]

df["cefr_level"] = df["cefr_level"].str.replace("+", "", regex=False)
df["target"] = df["cefr_level"].map(CEFR_MAP)
df = df.dropna(subset=["target"]).reset_index(drop=True)
y = df["target"].astype(int).values


def _lookup_exact(lower):
    entry = dictionary.get(lower)
    if not entry:
        return None
    if "l" in entry:
        base_word = entry["l"]
        base = dictionary.get(base_word)
        if not base:
            return (lower, 0.0, entry.get("c"))
        c = base["c"] if "c" in base else entry.get("c")
        return (base_word, base.get("z", 0.0), c)
    return (lower, entry.get("z", 0.0), entry.get("c"))


def resolve_entry(word):
    lower = word.lower().replace("’", "'")
    hit = _lookup_exact(lower)
    if hit:
        return hit
    if lower.endswith("'s") and len(lower) > 3:
        hit = _lookup_exact(lower[:-2])
        if hit:
            return hit
    if "-" in lower:
        hit = _lookup_exact(lower.replace("-", ""))
        if hit:
            return hit
    return None


def find_separable_matches(clause_words):
    matches = {}
    if len(clause_words) < 2:
        return matches

    last_idx = len(clause_words) - 1
    particle_lower = clause_words[last_idx].lower()
    if particle_lower not in SEPARABLE_PARTICLES:
        return matches

    for i in range(last_idx):
        raw_verb = clause_words[i]
        verb_lower = raw_verb.lower()
        if verb_lower in SEPARABLE_PARTICLES:
            continue

        verb_resolved = resolve_entry(raw_verb)
        stem_lemmas = []
        if verb_resolved and is_verb_infinitive(verb_resolved[0]):
            stem_lemmas.append(verb_resolved[0])

        for suf in ("ten", "den", "te", "de", "t"):
            if verb_lower.endswith(suf) and len(verb_lower) - len(suf) >= 3:
                root = verb_lower[:-len(suf)]
                stem_lemmas.append(root + "en")
                if len(root) >= 2 and root[-1] == root[-2]:
                    stem_lemmas.append(root[:-1] + "en")

        for stem in stem_lemmas:
            if stem in AUX_OR_STOP_VERBS:
                continue
            candidate = particle_lower + stem
            comp_hit = _lookup_exact(candidate)
            if comp_hit and (candidate in dictionary and ("d" in dictionary[candidate] or "c" in dictionary[candidate] or "l" in dictionary[candidate])):
                merged_len = len(raw_verb) + len(clause_words[last_idx])
                matches[i] = (comp_hit, merged_len, "verb")
                matches[last_idx] = (comp_hit, merged_len, "particle")
                return matches

    return matches


def split_sentence_clauses(sentence_str):
    return [words for c in CLAUSE_SPLIT_RE.split(sentence_str) if (words := WORD_RE.findall(c))]


def split_sentences(text):
    cleaned = text.strip().replace("’", "'")
    cleaned = ABBREV_RE.sub(lambda m: m.group(0).replace(".", "·"), cleaned)
    raw_sentences = [s.strip() for s in SENT_SPLIT_RE.split(cleaned) if s.strip()]
    return [clauses for s in raw_sentences if (clauses := split_sentence_clauses(s))]


def extract_features(text):
    sent_clause_lists = split_sentences(text)
    sent_lengths = []
    merged_tokens = []  # (word_len, resolved_tuple_or_None, raw_lower)

    for clauses in sent_clause_lists:
        sent_word_count = 0
        for clause_words in clauses:
            sep_matches = find_separable_matches(clause_words)
            for i, w in enumerate(clause_words):
                sep = sep_matches.get(i)
                if sep:
                    comp_hit, merged_len, role = sep
                    if role == "particle":
                        continue
                    sent_word_count += 1
                    merged_tokens.append((merged_len, comp_hit, comp_hit[0]))
                else:
                    sent_word_count += 1
                    merged_tokens.append((len(w), resolve_entry(w), w.lower()))
        if sent_word_count > 0:
            sent_lengths.append(sent_word_count)

    num_words = len(merged_tokens)
    num_sents = len(sent_lengths)
    if num_words == 0 or num_sents == 0:
        return [0.0] * len(FEATURES)

    avg_sent_len = num_words / num_sents
    max_sent_len = float(max(sent_lengths))
    std_sent_len = float(np.std(sent_lengths))
    avg_word_len = sum(t[0] for t in merged_tokens) / num_words
    long_word_ratio = sum(1 for t in merged_tokens if t[0] >= 10) / num_words

    unique_lemmas = set()
    valid_zipf = []
    nt2_first_scores = []
    rare_nt2_oov = 0
    common_nt2_oov = 0

    for _, resolved, raw_lower in merged_tokens:
        if resolved:
            lemma, z, c = resolved
            unique_lemmas.add(lemma)
            if z > 0:
                valid_zipf.append(z)
            if c is not None:
                nt2_first_scores.append(c)
            elif z > 0 and z >= RARE_ZIPF_THRESHOLD:
                common_nt2_oov += 1
            else:
                rare_nt2_oov += 1
        else:
            unique_lemmas.add(raw_lower)
            rare_nt2_oov += 1

    guiraud_ttr = len(unique_lemmas) / math.sqrt(num_words)

    if valid_zipf:
        p10_zipf = float(np.percentile(valid_zipf, 10))
        rare_word_ratio = sum(1 for z in valid_zipf if z < RARE_ZIPF_THRESHOLD) / len(valid_zipf)
    else:
        p10_zipf, rare_word_ratio = 0.0, 0.0

    nt2_mean_first = sum(nt2_first_scores) / len(nt2_first_scores) if nt2_first_scores else 4.0

    return [
        avg_sent_len, max_sent_len, std_sent_len,
        avg_word_len, long_word_ratio, guiraud_ttr,
        p10_zipf, rare_word_ratio,
        nt2_mean_first, rare_nt2_oov / num_words, common_nt2_oov / num_words,
    ]


def apply_thresholds(scores, thresholds):
    return np.digitize(scores, thresholds)


def fit_thresholds(y_true, raw_scores):
    init = [0.5, 1.5, 2.5, 3.5, 4.5]

    def loss(t):
        if any(t[i] >= t[i + 1] for i in range(len(t) - 1)):
            return 1.0
        preds = apply_thresholds(raw_scores, t)
        return -cohen_kappa_score(y_true, preds, weights="quadratic")

    res = minimize(loss, init, method="Nelder-Mead", options={"maxiter": 1000})
    return [round(float(x), 4) for x in res.x]


def build_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


print(f"2. Extracting 11 JS-parity features (v{MODEL_VERSION}) for {len(df)} texts...")
X = np.array([extract_features(t) for t in df["text"]], dtype=np.float32)

print("3. Validating naive vs. tuned thresholds (5-fold CV)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_raw = np.zeros(len(y), dtype=float)
oof_tuned = np.zeros(len(y), dtype=int)

for tr_idx, va_idx in skf.split(X, y):
    sw_tr = np.sqrt(compute_sample_weight("balanced", y[tr_idx]))
    m = build_model()
    m.fit(X[tr_idx], y[tr_idx], sample_weight=sw_tr)

    tr_scores = m.predict(X[tr_idx])
    va_scores = m.predict(X[va_idx])
    oof_raw[va_idx] = va_scores

    fold_t = fit_thresholds(y[tr_idx], tr_scores)
    oof_tuned[va_idx] = apply_thresholds(va_scores, fold_t)

naive_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
oof_naive = apply_thresholds(oof_raw, naive_thresholds)

for name, preds in [("Naive [0.5..4.5]", oof_naive), ("Tuned thresholds", oof_tuned)]:
    diffs = np.abs(y - preds)
    print(
        f"   {name:18s} | Exact: {np.mean(diffs == 0):.1%} | "
        f"Adj(±1): {np.mean(diffs <= 1):.1%} | Off(≥2): {np.mean(diffs >= 2):.1%} | "
        f"MAE: {mean_absolute_error(y, preds):.3f} | "
        f"QWK: {cohen_kappa_score(y, preds, weights='quadratic'):.3f} | "
        f"MacroRec: {recall_score(y, preds, average='macro'):.1%}"
    )

qwk_naive = cohen_kappa_score(y, oof_naive, weights="quadratic")
qwk_tuned = cohen_kappa_score(y, oof_tuned, weights="quadratic")
if qwk_tuned > qwk_naive + 0.003:
    final_thresholds = fit_thresholds(y, oof_raw)
    print(f"   -> Using tuned thresholds: {final_thresholds}")
else:
    final_thresholds = naive_thresholds
    print(f"   -> Keeping naive thresholds: {final_thresholds}")

print("4. Training final 11-feature (200-tree) model & exporting ONNX...")
sw_full = np.sqrt(compute_sample_weight("balanced", y))
final_model = build_model()
final_model.fit(X, y, sample_weight=sw_full)

initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(final_model, initial_types=initial_type)
with open(ONNX_OUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

meta = {
    "version": MODEL_VERSION,
    "model_type": "regressor",
    "features": FEATURES,
    "labels": CEFR_ORDER,
    "nt2_levels": NT2_LEVELS,
    "thresholds": final_thresholds,
    "rare_zipf_threshold": RARE_ZIPF_THRESHOLD,
}
with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"   Exported {ONNX_OUT} and {META_OUT}")

sample_text = (
    "Het NOS Journaal in Makkelijke Taal heeft minder onderwerpen dan de andere journaals. "
    "De presentator geeft meer uitleg en er worden geen moeilijke woorden gebruikt. "
    "Ook het tempo en het gebruik van beeld is rustiger. "
    "Kijkers van het eerdere weekjournaal (NOS Nieuws van de Week) vinden dat prettig, zo blijkt uit hun reacties."
)
sample_feats = np.array([extract_features(sample_text)], dtype=np.float32)
raw_score = float(final_model.predict(sample_feats)[0])
pred_label = CEFR_ORDER[int(apply_thresholds([raw_score], final_thresholds)[0])]

print("\n5. Python Parity Check (NOS Journaal snippet):")
for fname, val in zip(FEATURES, sample_feats[0]):
    print(f"   {fname:20s}: {val:.4f}")
print(f"   Raw Score           : {raw_score:.4f} -> {pred_label}")