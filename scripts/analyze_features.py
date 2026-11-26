import json
import math
import os
import re
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATA_PATH = "../data/local_elg_cefr_nl.json"
DICT_PATH = "../web/dictionary.json"
META_PATH = "../web/model_meta.json"
LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
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
            if comp_hit and (
                candidate in dictionary
                and (
                    "d" in dictionary[candidate]
                    or "c" in dictionary[candidate]
                    or "l" in dictionary[candidate]
                )
            ):
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
    merged_tokens = []

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
        row = {f: 0.0 for f in FEATURES}
        row["num_words"] = 0
        return row

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

    return {
        "avg_sent_len": avg_sent_len,
        "max_sent_len": max_sent_len,
        "std_sent_len": std_sent_len,
        "avg_word_len": avg_word_len,
        "long_word_ratio": long_word_ratio,
        "guiraud_ttr": guiraud_ttr,
        "p10_zipf": p10_zipf,
        "rare_word_ratio": rare_word_ratio,
        "nt2_mean_first": nt2_mean_first,
        "rare_nt2_oov_ratio": rare_nt2_oov / num_words,
        "common_nt2_oov_ratio": common_nt2_oov / num_words,
        "num_words": num_words,
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
    return [float(x) for x in res.x]


def build_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=200,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


print(f"2. Extracting 11 JS-parity features for {len(df)} texts...")
feat_df = pd.DataFrame([extract_features(t) for t in df["text"]])
X = feat_df[FEATURES].astype(np.float32)
df["num_words"] = feat_df["num_words"]

print("\n--- Dataset Overview ---")
for idx, lvl in enumerate(CEFR_ORDER):
    sub = df[df["target"] == idx]
    print(
        f"   {lvl}: {len(sub):4d} texts ({len(sub)/len(df):5.1%}) | "
        f"median words: {sub['num_words'].median():5.0f} | "
        f"min/max: {sub['num_words'].min():3d}/{sub['num_words'].max():4d}"
    )

print("\n3. Running 5-fold Stratified CV (learning curves + OOF predictions + OOF SHAP)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_raw = np.zeros(len(y), dtype=float)
oof_tuned = np.zeros(len(y), dtype=int)
oof_shap = np.zeros((len(y), len(FEATURES)), dtype=float)
train_rmse_folds = []
val_rmse_folds = []
fold_thresholds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    sw_tr = np.sqrt(compute_sample_weight("balanced", y_tr))
    sw_va = np.sqrt(compute_sample_weight("balanced", y_va))

    model = build_model()
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        sample_weight_eval_set=[sw_tr, sw_va],
        verbose=False,
    )

    evals = model.evals_result()
    train_rmse_folds.append(evals["validation_0"]["rmse"])
    val_rmse_folds.append(evals["validation_1"]["rmse"])

    tr_scores = model.predict(X_tr)
    va_scores = model.predict(X_va)
    oof_raw[va_idx] = va_scores

    t_fold = fit_thresholds(y_tr, tr_scores)
    fold_thresholds.append(t_fold)
    oof_tuned[va_idx] = apply_thresholds(va_scores, t_fold)

    explainer = shap.TreeExplainer(model)
    oof_shap[va_idx] = explainer.shap_values(X_va)

    fold_qwk = cohen_kappa_score(y_va, oof_tuned[va_idx], weights="quadratic")
    fold_mae = mean_absolute_error(y_va, oof_tuned[va_idx])
    print(f"   Fold {fold} | Val RMSE: {evals['validation_1']['rmse'][-1]:.4f} | MAE: {fold_mae:.3f} | QWK: {fold_qwk:.3f}")

diffs = np.abs(y - oof_tuned)
print("\n--- Overall 5-Fold OOF Metrics (Tuned Thresholds) ---")
print(
    f"   Exact: {np.mean(diffs == 0):.1%} | Adj(±1): {np.mean(diffs <= 1):.1%} | "
    f"Off(≥2): {np.mean(diffs >= 2):.1%} | MAE: {mean_absolute_error(y, oof_tuned):.3f} | "
    f"QWK: {cohen_kappa_score(y, oof_tuned, weights='quadratic'):.3f} | "
    f"Macro Recall: {recall_score(y, oof_tuned, average='macro'):.1%}"
)
print("\n--- Per-Class Classification Report (OOF) ---")
print(classification_report(y, oof_tuned, target_names=CEFR_ORDER, digits=3))

df["oof_raw"] = oof_raw
df["oof_pred"] = oof_tuned
df["abs_err"] = diffs
df["len_bucket"] = pd.cut(
    df["num_words"],
    bins=[0, 30, 80, 200, 500, 100000],
    labels=["<30", "30-80", "80-200", "200-500", "500+"],
)
print("--- Performance by Text Length (Words) ---")
for bucket, grp in df.groupby("len_bucket", observed=False):
    if len(grp) == 0:
        continue
    print(
        f"   {str(bucket):8s} (n={len(grp):4d}) | "
        f"Exact: {(grp['abs_err'] == 0).mean():5.1%} | "
        f"Adj(±1): {(grp['abs_err'] <= 1).mean():5.1%} | "
        f"Off(≥2): {(grp['abs_err'] >= 2).mean():5.1%} | "
        f"MAE: {grp['abs_err'].mean():.3f}"
    )

if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        final_thresholds = json.load(f).get("thresholds") or fit_thresholds(y, oof_raw)
else:
    final_thresholds = fit_thresholds(y, oof_raw)

sw_full = np.sqrt(compute_sample_weight("balanced", y))
final_model = build_model()
final_model.fit(X, y, sample_weight=sw_full)

print("\n4. Saving diagnostic plots to ../logs/...")

# --- Plot 1: Learning Curves ---
mean_tr_rmse = np.mean(train_rmse_folds, axis=0)
mean_va_rmse = np.mean(val_rmse_folds, axis=0)
std_va_rmse = np.std(val_rmse_folds, axis=0)
rounds = np.arange(1, len(mean_tr_rmse) + 1)
best_round = int(np.argmin(mean_va_rmse)) + 1

plt.figure(figsize=(9, 5))
plt.plot(rounds, mean_tr_rmse, label="Train RMSE (weighted)", color="#2563eb", lw=2)
plt.plot(rounds, mean_va_rmse, label="Val RMSE (weighted)", color="#dc2626", lw=2)
plt.fill_between(rounds, mean_va_rmse - std_va_rmse, mean_va_rmse + std_va_rmse, color="#dc2626", alpha=0.15)
plt.axvline(best_round, color="#0f172a", ls="--", lw=1.2, label=f"Best Val Round ({best_round}: {mean_va_rmse[best_round-1]:.3f})")
plt.xlabel("Boosting Round (Trees)")
plt.ylabel("Weighted RMSE")
plt.title("5-Fold CV Learning Curve (XGBRegressor, 11 Features)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/learning_curves.png", dpi=150)
plt.close()

# --- Plot 2: OOF Confusion Matrix ---
cm = confusion_matrix(y, oof_tuned, labels=range(6))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(CEFR_ORDER)
ax.set_yticklabels(CEFR_ORDER)
ax.set_xlabel("Predicted CEFR Level")
ax.set_ylabel("True CEFR Level")
ax.set_title("5-Fold OOF Confusion Matrix (Row % and Count)")

for i in range(6):
    for j in range(6):
        pct = cm_norm[i, j] * 100
        cnt = cm[i, j]
        color = "white" if cm_norm[i, j] > 0.5 else "#0f172a"
        ax.text(j, i, f"{pct:.1f}%\n({cnt})", ha="center", va="center", color=color, fontsize=9)

plt.colorbar(im, ax=ax, label="Row Proportion")
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/confusion_matrix_oof.png", dpi=150)
plt.close()

# --- Plot 3: Continuous OOF Score Distribution vs. Cutpoints ---
plt.figure(figsize=(9, 5.5))
grouped_scores = [oof_raw[y == i] for i in range(6)]
bp = plt.boxplot(grouped_scores, tick_labels=CEFR_ORDER, patch_artist=True, showfliers=True)
colors = ["#dcfce7", "#d1fae5", "#fef9c3", "#ffedd5", "#fee2e2", "#f3e8ff"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)

for t_val, boundary in zip(final_thresholds, ["A1|A2", "A2|B1", "B1|B2", "B2|C1", "C1|C2"]):
    plt.axhline(t_val, color="#dc2626", ls="--", lw=1, alpha=0.8)
    plt.text(6.55, t_val, f"{boundary} ({t_val:.2f})", va="center", fontsize=8, color="#991b1b")

plt.xlim(0.5, 7.2)
plt.xlabel("True CEFR Level")
plt.ylabel("Predicted Raw Score (OOF)")
plt.title("OOF Regression Score Distributions vs. Calibrated Cutpoints")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/score_distributions.png", dpi=150)
plt.close()

# --- Plot 4: Native XGBoost Gain Importance ---
fig, ax = plt.subplots(figsize=(9, 6))
xgb.plot_importance(
    final_model,
    ax=ax,
    importance_type="gain",
    title="XGBoost Feature Importance (Information Gain, 11 Features)",
    show_values=False,
)
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/xgboost_gain.png", dpi=150)
plt.close()

# --- Plot 5A: OOF SHAP Beeswarm ---
plt.figure(figsize=(10, 6.5))
shap.summary_plot(oof_shap, X, show=False)
plt.title("Out-of-Fold SHAP Values (11 Features, v1.5.1)")
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/shap_beeswarm.png", dpi=150)
plt.close()

# --- Plot 5B: OOF SHAP Global Bar Plot ---
plt.figure(figsize=(9, 5.5))
shap.summary_plot(oof_shap, X, plot_type="bar", show=False)
plt.title("Mean |SHAP| Impact on CEFR Score (5-Fold OOF, 11 Features)")
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/shap_bar.png", dpi=150)
plt.close()

# --- Error Analysis: Export Severe Errors (Off by >= 2 levels) ---
severe = df[df["abs_err"] >= 2].copy()
severe["true_label"] = severe["target"].map(lambda i: CEFR_ORDER[i])
severe["pred_label"] = severe["oof_pred"].map(lambda i: CEFR_ORDER[i])
severe["raw_err"] = np.abs(severe["target"] - severe["oof_raw"])
severe["text_preview"] = severe["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 140)
severe = severe.sort_values(by="raw_err", ascending=False)

err_cols = ["true_label", "pred_label", "oof_raw", "num_words", "text_preview"]
severe[err_cols].to_csv(f"{LOG_DIR}/severe_errors.csv", index=False)

print(f"\n5. Saved 6 plots + {len(severe)} severe errors to {LOG_DIR}/")
print("\n--- Top 8 Worst OOF Predictions (Off by ≥2 Levels) ---")
for _, r in severe.head(8).iterrows():
    print(
        f"   True {r['true_label']} -> Pred {r['pred_label']} (raw {r['oof_raw']:.2f}, "
        f"{int(r['num_words'])} words): {r['text_preview']}..."
    )