import csv
import json
from wordfreq import zipf_frequency

WIKT_PATH = "../data/kaikki.org-dictionary-Dutch.jsonl"
NT2_PATH = "../data/NT2Lex-CGN-v01.tsv"
OUTPUT_FILE = "../web/dictionary.json"
NT2_LEVELS = ["A1", "A2", "B1", "B2", "C1"]


def parse_num(val):
    if not val or val == "-":
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


print("1. Pass 1: Reading Wiktionary entries...")
base_defs = {}
lemma_links = {}

with open(WIKT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        word = data.get("word", "").strip().lower()

        if not word or " " in word or word.startswith("-") or word.endswith("-"):
            continue
        if not any(ch.isalpha() for ch in word):
            continue

        for sense in data.get("senses", []):
            form_of = sense.get("form_of") or sense.get("alt_of")
            glosses = sense.get("glosses", [])

            if form_of:
                target = form_of[0].get("word", "").strip().lower()
                if target and target != word and " " not in target:
                    targets = lemma_links.setdefault(word, [])
                    if target not in targets:
                        targets.append(target)
            elif glosses and word not in base_defs:
                clean_def = glosses[0].strip()
                if not clean_def.endswith(":"):
                    base_defs[word] = clean_def

print(f"   Found {len(base_defs)} base lemmas and {len(lemma_links)} words with inflection links.")

print("2. Pass 2: Reading NT2Lex-CGN-v01.tsv (first-appearance CEFR levels)...")
nt2_freq = {}
with open(NT2_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        word = (row.get("word") or "").strip().lower()
        if not word or " " in word or "[" in word or not any(ch.isalpha() for ch in word):
            continue
        if word not in nt2_freq:
            nt2_freq[word] = [0.0] * 5
        for i, lvl in enumerate(NT2_LEVELS):
            nt2_freq[word][i] += parse_num(row.get(f"F@{lvl}"))

nt2_first = {
    word: next(i for i, fv in enumerate(freqs) if fv > 0)
    for word, freqs in nt2_freq.items()
    if sum(freqs) > 0
}
print(f"   Extracted {len(nt2_first)} single-word NT2Lex lemmas.")


def resolve_base(start_word, first_target):
    seen = {start_word}
    curr = first_target
    while curr not in base_defs and curr in lemma_links and curr not in seen:
        seen.add(curr)
        curr = lemma_links[curr][0]
    return curr if (curr in base_defs and curr != start_word) else None


print("3. Merging definitions, Zipf scores, lemma pointers, and homograph-safe NT2Lex levels...")
mini_dict = {}

# Base lemmas: inherit the lowest (easiest) NT2Lex level across direct + linked lemma senses
for word, definition in base_defs.items():
    obj = {
        "d": definition,
        "z": round(zipf_frequency(word, "nl"), 2),
    }
    candidates = []
    if word in nt2_first:
        candidates.append(nt2_first[word])
    for target in lemma_links.get(word, []):
        if target in nt2_first:
            candidates.append(nt2_first[target])
        base_target = resolve_base(word, target)
        if base_target and base_target in nt2_first:
            candidates.append(nt2_first[base_target])
    if candidates:
        obj["c"] = min(candidates)
    mini_dict[word] = obj

# Single-hop inflection pointers
linked_count = 0
for word, targets in lemma_links.items():
    if word in mini_dict:
        continue

    curr = next((b for t in targets if (b := resolve_base(word, t))), None)
    if curr:
        obj = {"l": curr}
        if word in nt2_first and "c" not in mini_dict[curr]:
            obj["c"] = nt2_first[word]
        mini_dict[word] = obj
        linked_count += 1

# NT2Lex-only lemmas not present in Wiktionary (CEFR level only, no d/z)
nt2_only_count = 0
for word, lvl in nt2_first.items():
    if word not in mini_dict:
        mini_dict[word] = {"c": lvl}
        nt2_only_count += 1

# Integrity guard before writing production asset
assert all(
    ("d" in v and "z" in v)
    or ("l" in v and "z" in mini_dict[v["l"]])
    or ("c" in v and "l" not in v and "d" not in v)
    for v in mini_dict.values()
), "Invalid entry detected in dictionary!"

# Quick homograph sanity check
for check_word in ("meer", "was", "kan", "wil"):
    entry = mini_dict.get(check_word, {})
    lvl = NT2_LEVELS[entry["c"]] if "c" in entry else "None"
    print(f"   Homograph check '{check_word}': level={lvl}, z={entry.get('z')}")

print(
    f"4. Saving {len(mini_dict)} entries "
    f"({len(base_defs)} base + {linked_count} pointers + {nt2_only_count} NT2-only)..."
)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(mini_dict, f, ensure_ascii=False, separators=(",", ":"))

print(f"Done! Saved directly to {OUTPUT_FILE}")