import json
import math
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATA_PATH = "../data/local_elg_cefr_nl.json"
DICT_PATH = "../web/dictionary.json"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_MAP = {lvl: i for i, lvl in enumerate(CEFR_ORDER)}

SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
RARE_ZIPF_THRESHOLD = 4.0

print("1. Loading dictionary.json and dataset...")
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


def resolve_entry(word):
    lower = word.lower()
    entry = dictionary.get(lower)
    if not entry:
        return None
    if "l" in entry:
        base_word = entry["l"]
        base = dictionary.get(base_word)
        return (base_word, base["z"]) if base else None
    return (lower, entry["z"])


def extract_js_parity_features(text):
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
        return {
            "avg_sent_len": 0.0, "max_sent_len": 0.0, "std_sent_len": 0.0,
            "avg_word_len": 0.0, "long_word_ratio": 0.0,
            "ttr": 0.0, "guiraud_ttr": 0.0,
            "avg_zipf": 0.0, "p10_zipf": 0.0, "rare_word_ratio": 0.0,
        }

    avg_sent_len = num_words / num_sents
    max_sent_len = float(max(sent_lengths))
    std_sent_len = float(np.std(sent_lengths))
    avg_word_len = sum(len(w) for w in all_words) / num_words
    long_word_ratio = sum(1 for w in all_words if len(w) >= 10) / num_words

    unique_lemmas = set()
    valid_zipf = []

    for w in all_words:
        resolved = resolve_entry(w)
        if resolved:
            lemma, z = resolved
            unique_lemmas.add(lemma)
            if z > 0:
                valid_zipf.append(z)
        else:
            unique_lemmas.add(w.lower())

    ttr = len(unique_lemmas) / num_words
    guiraud_ttr = len(unique_lemmas) / math.sqrt(num_words)

    if valid_zipf:
        avg_zipf = sum(valid_zipf) / len(valid_zipf)
        p10_zipf = float(np.percentile(valid_zipf, 10))
        rare_word_ratio = sum(1 for z in valid_zipf if z < RARE_ZIPF_THRESHOLD) / len(valid_zipf)
    else:
        avg_zipf, p10_zipf, rare_word_ratio = 0.0, 0.0, 0.0

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
    }


print(f"2. Extracting JS-parity features for {len(df)} texts...")
X_full = pd.DataFrame([extract_js_parity_features(t) for t in df["text"]])

FEATURE_SETS = {
    "Lean_6": [
        "avg_sent_len", "max_sent_len", "avg_word_len",
        "ttr", "avg_zipf", "rare_word_ratio"
    ],
    "Enriched_10": [
        "avg_sent_len", "max_sent_len", "std_sent_len",
        "avg_word_len", "long_word_ratio", "ttr", "guiraud_ttr",
        "avg_zipf", "p10_zipf", "rare_word_ratio"
    ],
}


def get_models():
    return [
        ("XGBClassifier (default, balanced)", "clf", "balanced",
         lambda: xgb.XGBClassifier(
             objective="multi:softmax", num_class=6, eval_metric="mlogloss",
             max_depth=5, subsample=0.8, random_state=42
         )),
        ("XGBClassifier (reg, balanced)", "clf", "balanced",
         lambda: xgb.XGBClassifier(
             objective="multi:softmax", num_class=6, eval_metric="mlogloss",
             n_estimators=250, learning_rate=0.04, max_depth=4,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42
         )),
        ("XGBRegressor (reg, unweighted)", "reg", "none",
         lambda: xgb.XGBRegressor(
             objective="reg:squarederror",
             n_estimators=250, learning_rate=0.04, max_depth=4,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42
         )),
        ("XGBRegressor (reg, sqrt-bal)", "reg", "sqrt",
         lambda: xgb.XGBRegressor(
             objective="reg:squarederror",
             n_estimators=250, learning_rate=0.04, max_depth=4,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42
         )),
        ("XGBRegressor (reg, balanced)", "reg", "balanced",
         lambda: xgb.XGBRegressor(
             objective="reg:squarederror",
             n_estimators=250, learning_rate=0.04, max_depth=4,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42
         )),
        ("HistGBRegressor (sqrt-bal)", "reg", "sqrt",
         lambda: HistGradientBoostingRegressor(
             max_depth=4, learning_rate=0.05, max_iter=200,
             min_samples_leaf=20, random_state=42
         )),
        ("Ridge (linear, sqrt-bal)", "reg_scaled", "sqrt",
         lambda: Ridge(alpha=10.0)),
    ]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

print("3. Running 5-fold Stratified CV benchmark...\n")
for feat_name, cols in FEATURE_SETS.items():
    X_mat = X_full[cols].values

    for model_name, kind, weight_mode, builder in get_models():
        oof_preds = np.zeros(len(y), dtype=int)

        for train_idx, val_idx in skf.split(X_mat, y):
            X_tr, X_va = X_mat[train_idx], X_mat[val_idx]
            y_tr = y[train_idx]

            if weight_mode == "balanced":
                sw = compute_sample_weight("balanced", y_tr)
            elif weight_mode == "sqrt":
                sw = np.sqrt(compute_sample_weight("balanced", y_tr))
            else:
                sw = None

            if kind == "reg_scaled":
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)

            m = builder()
            if sw is not None:
                m.fit(X_tr, y_tr, sample_weight=sw)
            else:
                m.fit(X_tr, y_tr)

            raw = m.predict(X_va)
            if kind == "clf":
                oof_preds[val_idx] = raw.astype(int)
            else:
                oof_preds[val_idx] = np.clip(np.rint(raw), 0, 5).astype(int)

        diffs = np.abs(y - oof_preds)
        rows.append({
            "Features": feat_name,
            "Model": model_name,
            "Exact": f"{np.mean(diffs == 0):.1%}",
            "Adj(±1)": f"{np.mean(diffs <= 1):.1%}",
            "Off(≥2)": f"{np.mean(diffs >= 2):.1%}",
            "MAE": round(mean_absolute_error(y, oof_preds), 3),
            "QWK": round(cohen_kappa_score(y, oof_preds, weights="quadratic"), 3),
            "MacroRec": f"{recall_score(y, oof_preds, average='macro'):.1%}",
        })

results_df = pd.DataFrame(rows).sort_values(by=["QWK", "MAE"], ascending=[False, True])
print(results_df.to_string(index=False))