import csv
import json
import math
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATA_PATH = "../data/local_elg_cefr_nl.json"
DICT_PATH = "../web/dictionary.json"
NT2_PATH = "../data/NT2Lex-CGN-v01.tsv"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_MAP = {lvl: i for i, lvl in enumerate(CEFR_ORDER)}
NT2_LEVELS = ["A1", "A2", "B1", "B2", "C1"]

SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
RARE_ZIPF_THRESHOLD = 4.0


def parse_num(val):
    if not val or val == "-":
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


print("1. Loading dictionary.json and NT2Lex-CGN-v01.tsv...")
with open(DICT_PATH, "r", encoding="utf-8") as f:
    dictionary = json.load(f)

# Aggregate NT2Lex across POS tags per single-word lemma
nt2_raw = {}
with open(NT2_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for lvl in NT2_LEVELS:
        assert f"F@{lvl}" in reader.fieldnames and f"U@{lvl}" in reader.fieldnames, (
            f"Missing expected columns for {lvl}. Found: {reader.fieldnames}"
        )

    for row in reader:
        word = (row.get("word") or "").strip().lower()
        if not word or " " in word or "[" in word or not any(ch.isalpha() for ch in word):
            continue

        if word not in nt2_raw:
            nt2_raw[word] = {"F": [0.0] * 5, "U": [0.0] * 5}

        for i, lvl in enumerate(NT2_LEVELS):
            nt2_raw[word]["F"][i] += parse_num(row.get(f"F@{lvl}"))
            nt2_raw[word]["U"][i] += parse_num(row.get(f"U@{lvl}"))

# Summarize each NT2Lex lemma into (first_level, u_weighted_level)
nt2_lex = {}
for word, stats in nt2_raw.items():
    f_vals = stats["F"]
    u_vals = stats["U"]
    if sum(f_vals) <= 0:
        continue

    first_lvl = next(i for i, fv in enumerate(f_vals) if fv > 0)
    u_sum = sum(u_vals)
    weighted_lvl = (
        sum(i * uv for i, uv in enumerate(u_vals)) / u_sum
        if u_sum > 0 else float(first_lvl)
    )
    nt2_lex[word] = (first_lvl, weighted_lvl)

overlap = sum(1 for w in nt2_lex if w in dictionary)
print(f"   Loaded {len(nt2_lex)} single-word NT2Lex lemmas ({overlap} already in dictionary.json).")

df = pd.read_json(DATA_PATH)
df = df.dropna(subset=["cefr_level", "text"])
if "category" in df.columns:
    df = df[df["category"] == "reference"]

df["cefr_level"] = df["cefr_level"].str.replace("+", "", regex=False)
df["target"] = df["cefr_level"].map(CEFR_MAP)
df = df.dropna(subset=["target"]).reset_index(drop=True)
y = df["target"].astype(int).values


def resolve_entry(word):
    lower = word.lower()
    entry = dictionary.get(lower)
    if not entry:
        return (lower, 0.0)
    if "l" in entry:
        base_word = entry["l"]
        base = dictionary.get(base_word)
        return (base_word, base["z"]) if base else (lower, 0.0)
    return (lower, entry["z"])


def extract_all_features(text):
    raw_sentences = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    all_words = []
    sent_lengths = []

    for sent in raw_sentences:
        words = WORD_RE.findall(sent)
        if words:
            sent_lengths.append(len(words))
            all_words.extend(words)

    num_words = len(all_words)
    num_sents = len(sent_lengths)
    if num_words == 0 or num_sents == 0:
        return {k: 0.0 for k in ALL_COLS}

    avg_sent_len = num_words / num_sents
    max_sent_len = float(max(sent_lengths))
    std_sent_len = float(np.std(sent_lengths))
    avg_word_len = sum(len(w) for w in all_words) / num_words
    long_word_ratio = sum(1 for w in all_words if len(w) >= 10) / num_words

    unique_lemmas = set()
    valid_zipf = []
    nt2_first_scores = []
    nt2_weighted_scores = []
    nt2_a1a2 = 0
    nt2_b2plus = 0
    nt2_oov = 0

    for w in all_words:
        lemma, z = resolve_entry(w)
        unique_lemmas.add(lemma)
        if z > 0:
            valid_zipf.append(z)

        nt2_info = nt2_lex.get(lemma) or nt2_lex.get(w.lower())
        if nt2_info is not None:
            first_lvl, w_lvl = nt2_info
            nt2_first_scores.append(first_lvl)
            nt2_weighted_scores.append(w_lvl)
            if first_lvl <= 1:
                nt2_a1a2 += 1
            elif first_lvl >= 3:
                nt2_b2plus += 1
        else:
            nt2_oov += 1

    ttr = len(unique_lemmas) / num_words
    guiraud_ttr = len(unique_lemmas) / math.sqrt(num_words)

    if valid_zipf:
        avg_zipf = sum(valid_zipf) / len(valid_zipf)
        p10_zipf = float(np.percentile(valid_zipf, 10))
        rare_word_ratio = sum(1 for z in valid_zipf if z < RARE_ZIPF_THRESHOLD) / len(valid_zipf)
    else:
        avg_zipf, p10_zipf, rare_word_ratio = 0.0, 0.0, 0.0

    nt2_mean_first = sum(nt2_first_scores) / len(nt2_first_scores) if nt2_first_scores else 4.0
    nt2_p90_first = float(np.percentile(nt2_first_scores, 90)) if nt2_first_scores else 4.0
    nt2_mean_weighted = sum(nt2_weighted_scores) / len(nt2_weighted_scores) if nt2_weighted_scores else 4.0

    return {
        "avg_sent_len": avg_sent_len,
        "max_sent_len": max_sent_len,
        "std_sent_len": std_sent_len,
        "avg_word_len": avg_word_len,
        "long_word_ratio": long_word_ratio,
        "ttr": ttr,
        "guiraud_ttr": guiraud_ttr,
        "avg_zipf": avg_zipf,
        "p10_zipf": p10_zipf,
        "rare_word_ratio": rare_word_ratio,
        "nt2_mean_first": nt2_mean_first,
        "nt2_p90_first": nt2_p90_first,
        "nt2_a1a2_ratio": nt2_a1a2 / num_words,
        "nt2_b2plus_ratio": nt2_b2plus / num_words,
        "nt2_oov_ratio": nt2_oov / num_words,
        "nt2_mean_weighted": nt2_mean_weighted,
    }


BASE_10 = [
    "avg_sent_len", "max_sent_len", "std_sent_len",
    "avg_word_len", "long_word_ratio", "ttr", "guiraud_ttr",
    "avg_zipf", "p10_zipf", "rare_word_ratio",
]
NT2_FIRST_COLS = ["nt2_mean_first", "nt2_p90_first", "nt2_a1a2_ratio", "nt2_b2plus_ratio", "nt2_oov_ratio"]
ALL_COLS = BASE_10 + NT2_FIRST_COLS + ["nt2_mean_weighted"]

FEATURE_SETS = {
    "Baseline (Enriched_10)": BASE_10,
    "+ NT2 First-Level (15)": BASE_10 + NT2_FIRST_COLS,
    "+ NT2 All (16)": ALL_COLS,
    "NT2 Only + Struct (13)": BASE_10[:7] + NT2_FIRST_COLS + ["nt2_mean_weighted"],
}

print(f"2. Extracting features for {len(df)} texts...")
X_df = pd.DataFrame([extract_all_features(t) for t in df["text"]])
print(f"   Average token coverage in NT2Lex: {(1.0 - X_df['nt2_oov_ratio'].mean()):.1%}")


def apply_thresholds(scores, thresholds):
    return np.digitize(scores, thresholds)


def fit_thresholds(y_true, raw_scores):
    init = [0.5, 1.5, 2.5, 3.5, 4.5]

    def loss(t):
        if any(t[i] >= t[i + 1] for i in range(len(t) - 1)):
            return 1.0
        return -cohen_kappa_score(y_true, apply_thresholds(raw_scores, t), weights="quadratic")

    res = minimize(loss, init, method="Nelder-Mead", options={"maxiter": 1000})
    return res.x


def build_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250, learning_rate=0.04, max_depth=4,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42,
    )


print("\n3. Running 5-fold Stratified CV (XGBRegressor, sqrt-bal, tuned thresholds)...\n")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, cols in FEATURE_SETS.items():
    X_mat = X_df[cols].values.astype(np.float32)
    oof_naive = np.zeros(len(y), dtype=int)
    oof_tuned = np.zeros(len(y), dtype=int)
    importances = np.zeros(len(cols))

    for tr_idx, va_idx in skf.split(X_mat, y):
        sw_tr = np.sqrt(compute_sample_weight("balanced", y[tr_idx]))
        m = build_model()
        m.fit(X_mat[tr_idx], y[tr_idx], sample_weight=sw_tr)

        tr_scores = m.predict(X_mat[tr_idx])
        va_scores = m.predict(X_mat[va_idx])
        oof_naive[va_idx] = apply_thresholds(va_scores, [0.5, 1.5, 2.5, 3.5, 4.5])
        oof_tuned[va_idx] = apply_thresholds(va_scores, fit_thresholds(y[tr_idx], tr_scores))
        importances += m.feature_importances_ / 5.0

    for mode, preds in [("Naive", oof_naive), ("Tuned", oof_tuned)]:
        diffs = np.abs(y - preds)
        rows.append({
            "Feature Set": name,
            "Thresholds": mode,
            "Exact": f"{np.mean(diffs == 0):.1%}",
            "Adj(±1)": f"{np.mean(diffs <= 1):.1%}",
            "Off(≥2)": f"{np.mean(diffs >= 2):.1%}",
            "MAE": round(mean_absolute_error(y, preds), 3),
            "QWK": round(cohen_kappa_score(y, preds, weights="quadratic"), 3),
            "MacroRec": f"{recall_score(y, preds, average='macro'):.1%}",
        })

    if name == "+ NT2 All (16)":
        top_feats = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)

print(pd.DataFrame(rows).sort_values(by="QWK", ascending=False).to_string(index=False))

print("\n4. Top Feature Importances (+ NT2 All):")
for fname, imp in top_feats:
    print(f"   {fname:18s}: {imp:.4f}")