import json
import math
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, recall_score, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATA_PATH = "../data/local_elg_cefr_nl.json"
DICT_PATH = "../web/dictionary.json"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_MAP = {lvl: i for i, lvl in enumerate(CEFR_ORDER)}

FULL_15 = [
    "avg_sent_len", "max_sent_len", "std_sent_len",
    "avg_word_len", "long_word_ratio", "ttr", "guiraud_ttr",
    "avg_zipf", "p10_zipf", "rare_word_ratio",
    "nt2_mean_first", "nt2_p90_first", "nt2_a1a2_ratio",
    "nt2_b2plus_ratio", "nt2_oov_ratio",
]

PRUNED_10 = [
    "avg_sent_len", "max_sent_len", "std_sent_len",
    "avg_word_len", "long_word_ratio", "guiraud_ttr",
    "p10_zipf", "rare_word_ratio",
    "nt2_mean_first", "nt2_oov_ratio",
]

# Monotone direction for PRUNED_10: +1 harder, -1 easier, 0 unconstrained
MONOTONE_10 = (
    1,   # avg_sent_len
    0,   # max_sent_len
    0,   # std_sent_len
    1,   # avg_word_len
    1,   # long_word_ratio
    0,   # guiraud_ttr
    -1,  # p10_zipf (lower Zipf = rarer = harder)
    1,   # rare_word_ratio
    1,   # nt2_mean_first
    0,   # nt2_oov_ratio (proper nouns can be OOV even in A1/A2)
)

SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
RARE_ZIPF_THRESHOLD = 4.0

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


def resolve_entry(word):
    lower = word.lower()
    entry = dictionary.get(lower)
    if not entry:
        return (lower, 0.0, None)
    if "l" in entry:
        base_word = entry["l"]
        base = dictionary.get(base_word)
        if not base:
            return (lower, 0.0, entry.get("c"))
        c = base["c"] if "c" in base else entry.get("c")
        return (base_word, base.get("z", 0.0), c)
    return (lower, entry.get("z", 0.0), entry.get("c"))


def extract_all(text):
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
        return {k: 0.0 for k in FULL_15}

    avg_sent_len = num_words / num_sents
    max_sent_len = float(max(sent_lengths))
    std_sent_len = float(np.std(sent_lengths))
    avg_word_len = sum(len(w) for w in all_words) / num_words
    long_word_ratio = sum(1 for w in all_words if len(w) >= 10) / num_words

    unique_lemmas = set()
    valid_zipf = []
    nt2_first_scores = []
    nt2_a1a2 = 0
    nt2_b2plus = 0
    nt2_oov = 0

    for w in all_words:
        lemma, z, c = resolve_entry(w)
        unique_lemmas.add(lemma)
        if z > 0:
            valid_zipf.append(z)
        if c is not None:
            nt2_first_scores.append(c)
            if c <= 1:
                nt2_a1a2 += 1
            elif c >= 3:
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
    }


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


print(f"2. Extracting features for {len(df)} texts...")
X_df = pd.DataFrame([extract_all(t) for t in df["text"]])

CONFIGS = [
    ("15-feat (250 trees)", FULL_15, 250, None),
    ("15-feat (200 trees)", FULL_15, 200, None),
    ("Pruned 10-feat (200 trees)", PRUNED_10, 200, None),
    ("Pruned 10-feat + Mono (200)", PRUNED_10, 200, MONOTONE_10),
]

SAMPLES = {
    "A2 (NOS, 53w)": (
        "Het NOS Journaal in Makkelijke Taal heeft minder onderwerpen dan de andere journaals. "
        "De presentator geeft meer uitleg en er worden geen moeilijke woorden gebruikt. "
        "Ook het tempo en het gebruik van beeld is rustiger. "
        "Kijkers van het eerdere weekjournaal (NOS Nieuws van de Week) vinden dat prettig, zo blijkt uit hun reacties."
    ),
    "B2 (News, 52w)": (
        "Nederlandse gemeenten maken zich steeds meer zorgen over het tekort aan betaalbare huurwoningen. "
        "Hoewel het kabinet vorig jaar nieuwe maatregelen aankondigde om de woningbouw te versnellen, "
        "lopen veel projecten vertraging op door gestegen bouwkosten en stikstofregels. "
        "Woningcorporaties pleiten daarom voor snellere vergunningsprocedures en meer financiële steun vanuit Den Haag."
    ),
    "C1 (Policy, 57w)": (
        "De toenemende verwevenheid van geopolitieke belangen en mondiale toeleveringsketens dwingt Europese beleidsmakers "
        "tot een fundamentele heroverweging van hun industriestrategie. Waar decennialang werd uitgegaan van ongehinderde "
        "marktintegratie als waarborg voor welvaart, verschuift de nadruk thans naar strategische autonomie en weerbaarheid. "
        "Deze koerswijziging roept echter complexe vraagstukken op omtrent mededinging, subsidies en de verdeling van maatschappelijke lasten."
    ),
    "C1 Short (34w)": (
        "De toenemende verwevenheid van geopolitieke belangen en mondiale toeleveringsketens dwingt Europese beleidsmakers "
        "tot een fundamentele heroverweging van hun industriestrategie, waarbij de nadruk verschuift naar strategische autonomie, "
        "economische weerbaarheid en rechtvaardige verdeling van maatschappelijke lasten."
    ),
}
sample_df = pd.DataFrame([extract_all(t) for t in SAMPLES.values()], index=SAMPLES.keys())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
sample_preds = {}

print("3. Running 5-fold Stratified CV comparison...\n")
for name, cols, n_est, mono in CONFIGS:
    X_mat = X_df[cols].values.astype(np.float32)
    oof_raw = np.zeros(len(y), dtype=float)
    oof_tuned = np.zeros(len(y), dtype=int)
    w_rmses = []

    params = dict(
        objective="reg:squarederror",
        n_estimators=n_est,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    if mono is not None:
        params["monotone_constraints"] = mono

    for tr_idx, va_idx in skf.split(X_mat, y):
        sw_tr = np.sqrt(compute_sample_weight("balanced", y[tr_idx]))
        sw_va = np.sqrt(compute_sample_weight("balanced", y[va_idx]))

        m = xgb.XGBRegressor(**params)
        m.fit(X_mat[tr_idx], y[tr_idx], sample_weight=sw_tr)

        tr_scores = m.predict(X_mat[tr_idx])
        va_scores = m.predict(X_mat[va_idx])
        oof_raw[va_idx] = va_scores
        w_rmses.append(root_mean_squared_error(y[va_idx], va_scores, sample_weight=sw_va))
        oof_tuned[va_idx] = apply_thresholds(va_scores, fit_thresholds(y[tr_idx], tr_scores))

    oof_naive = apply_thresholds(oof_raw, [0.5, 1.5, 2.5, 3.5, 4.5])
    c1_rec = recall_score(y, oof_tuned, average=None)[4]

    for mode, preds in [("Tuned", oof_tuned), ("Naive", oof_naive)]:
        diffs = np.abs(y - preds)
        rows.append({
            "Model": name,
            "Thresh": mode,
            "wRMSE": round(float(np.mean(w_rmses)), 4),
            "Exact": f"{np.mean(diffs == 0):.1%}",
            "Adj(±1)": f"{np.mean(diffs <= 1):.1%}",
            "Off(≥2)": f"{np.mean(diffs >= 2):.1%}",
            "MAE": round(mean_absolute_error(y, preds), 3),
            "QWK": round(cohen_kappa_score(y, preds, weights="quadratic"), 3),
            "MacroRec": f"{recall_score(y, preds, average='macro'):.1%}",
            "C1_Rec": f"{c1_rec:.1%}" if mode == "Tuned" else "-",
        })

    # Fit full model to inspect predictions on short real-world texts
    sw_full = np.sqrt(compute_sample_weight("balanced", y))
    full_m = xgb.XGBRegressor(**params)
    full_m.fit(X_mat, y, sample_weight=sw_full)
    t_full = fit_thresholds(y, oof_raw)

    s_scores = full_m.predict(sample_df[cols].values.astype(np.float32))
    s_labels = [CEFR_ORDER[int(i)] for i in apply_thresholds(s_scores, t_full)]
    sample_preds[name] = [f"{lbl} ({sc:.2f})" for lbl, sc in zip(s_labels, s_scores)]

print(pd.DataFrame(rows).sort_values(by=["Thresh", "QWK"], ascending=[False, False]).to_string(index=False))

print("\n4. Short-Text Real-World Stress Test (Full Model + Tuned Thresholds):")
stress_df = pd.DataFrame(sample_preds, index=SAMPLES.keys())
print(stress_df.to_string())