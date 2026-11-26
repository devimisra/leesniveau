export const ABBREV_RE =
  /\b(?:o\.a|bijv|d\.w\.z|i\.p\.v|m\.b\.v|m\.a\.w|t\.o\.v|t\.a\.v|i\.v\.m|n\.a\.v|o\.b\.v|o\.i\.d|e\.d|enz|etc|nl|dr|drs|prof|mr|ir|dhr|mw|mevr|fam|blz|pag|nr|ca|st|max|min|tel|a\.u\.b|s\.v\.p)\./gi;
export const SENT_SPLIT_RE = /[.!?]+(?:\s+|$)|\r?\n+/;
export const WORD_RE = /[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*/g;
export const LONG_TEXT_WORD_LIMIT = 350;
export const CHUNK_TARGET_WORDS = 200;

// Safe dictionary lookup (guards against JS Object.prototype keys like "constructor")
export function lookupExact(dictionary, lower) {
  if (!dictionary || !Object.prototype.hasOwnProperty.call(dictionary, lower)) return null;
  const entry = dictionary[lower];
  if (!entry) return null;

  if (entry.l) {
    const baseWord = entry.l;
    const base = Object.prototype.hasOwnProperty.call(dictionary, baseWord)
      ? dictionary[baseWord]
      : null;
    if (!base) {
      return {
        lemma: lower,
        d: null,
        z: 0,
        c: entry.c !== undefined ? entry.c : null
      };
    }
    const c = base.c !== undefined ? base.c : (entry.c !== undefined ? entry.c : null);
    return {
      lemma: baseWord,
      d: base.d !== undefined ? base.d : null,
      z: base.z !== undefined ? base.z : 0,
      c
    };
  }

  return {
    lemma: lower,
    d: entry.d !== undefined ? entry.d : null,
    z: entry.z !== undefined ? entry.z : 0,
    c: entry.c !== undefined ? entry.c : null
  };
}

// Resolve inflections, 's plurals (auto's -> auto), and hyphenated forms (e-mail -> email)
export function resolveEntry(dictionary, rawWord) {
  const lower = rawWord.toLowerCase().replace(/’/g, "'");
  let hit = lookupExact(dictionary, lower);
  if (hit) return hit;

  if (lower.endsWith("'s") && lower.length > 3) {
    hit = lookupExact(dictionary, lower.slice(0, -2));
    if (hit) return hit;
  }
  if (lower.includes("-")) {
    hit = lookupExact(dictionary, lower.replace(/-/g, ""));
    if (hit) return hit;
  }
  return null;
}

// UI-only helper: resolve inflected Dutch adjectives/comparatives/superlatives (snellere -> snel, langjarige -> langjarig)
export function resolveInflectedUi(dictionary, rawWord) {
  const lower = rawWord.toLowerCase().replace(/’/g, "'");
  if (lower.length < 5) return null;

  const vowels = "aeiouy";
  const candidates = [];

  for (const suf of ["ere", "ste", "er", "st", "e"]) {
    if (!lower.endsWith(suf) || lower.length - suf.length < 3) continue;
    const base = lower.slice(0, -suf.length);
    candidates.push(base);

    if (base.length >= 4 && base.at(-1) === base.at(-2) && !vowels.includes(base.at(-1))) {
      candidates.push(base.slice(0, -1));
    }

    if (
      base.length >= 3 &&
      !vowels.includes(base.at(-1)) &&
      "aeou".includes(base.at(-2)) &&
      !vowels.includes(base.at(-3))
    ) {
      candidates.push(base.slice(0, -1) + base.at(-2) + base.at(-1));
    }

    if (base.endsWith("v")) {
      candidates.push(base.slice(0, -1) + "f");
    } else if (base.endsWith("z")) {
      candidates.push(base.slice(0, -1) + "s");
      if (base.length >= 3 && "aeou".includes(base.at(-2)) && !vowels.includes(base.at(-3))) {
        candidates.push(base.slice(0, -2) + base.at(-2) + "s");
      }
    }

    if (base.includes("ë")) {
      const clean = base.replace(/ë/g, "e");
      candidates.push(clean);
      if (clean.endsWith("iel")) candidates.push(clean.slice(0, -3) + "ieel");
    }
  }

  for (const cand of candidates) {
    if (cand === lower) continue;
    const hit = resolveEntry(dictionary, cand);
    if (hit && hit.d) return hit;
  }
  return null;
}

// Split unlisted Dutch compounds & derivational prefixes (e.g., weekjournaal, heroverweging)
export function splitCompound(dictionary, rawWord) {
  if (rawWord.length < 6 || /^[A-Z0-9]+$/.test(rawWord)) return null;

  const lower = rawWord.toLowerCase();
  const prefixes = [
    ["her", "re-"],
    ["wan", "mis- / bad"],
    ["mis", "mis- / wrong"]
  ];
  for (const [pref, gloss] of prefixes) {
    if (lower.startsWith(pref) && lower.length - pref.length >= 4) {
      const stem = resolveEntry(dictionary, lower.slice(pref.length));
      if (stem && stem.d && stem.z >= 2.5) {
        return {
          lemma: lower,
          d: `${pref}- (${gloss}) + ${stem.lemma} (${stem.d})`,
          z: stem.z,
          c: null,
          isCompound: true
        };
      }
    }
  }

  const linkers = ["", "s", "e", "en"];
  let best = null;

  for (let i = 3; i <= lower.length - 3; i++) {
    const leftRes = resolveEntry(dictionary, lower.slice(0, i));
    if (!leftRes || !leftRes.d || leftRes.z < 2.5) continue;

    for (const linker of linkers) {
      if (linker && !lower.startsWith(linker, i)) continue;
      const rightRaw = lower.slice(i + linker.length);
      if (rightRaw.length < 3) continue;

      const rightRes = resolveEntry(dictionary, rightRaw);
      if (!rightRes || !rightRes.d || rightRes.z < 2.5) continue;

      const score = leftRes.z + rightRes.z + (linker === "" ? 0.2 : 0);
      if (!best || score > best.score) {
        best = {
          lemma: lower,
          d: `${leftRes.lemma} (${leftRes.d}) + ${rightRes.lemma} (${rightRes.d})`,
          z: Math.min(leftRes.z, rightRes.z),
          c: null,
          isCompound: true,
          score
        };
      }
    }
  }

  return best;
}

// Matches NumPy's default linear-interpolated percentile
export function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// Matches np.digitize(score, thresholds)
export function applyThresholds(score, thresholds) {
  let idx = 0;
  while (idx < thresholds.length && score >= thresholds[idx]) {
    idx++;
  }
  return idx;
}

// Flag scores within ±0.15 of a calibrated CEFR boundary
export function getBorderlineNote(rawScore, thresholds, labels) {
  if (!Array.isArray(thresholds) || !Array.isArray(labels)) return "";
  for (let i = 0; i < thresholds.length; i++) {
    if (Math.abs(rawScore - thresholds[i]) <= 0.15) {
      const lo = labels[i];
      const hi = labels[i + 1];
      if (lo && hi) return ` · borderline ${lo} / ${hi}`;
    }
  }
  return "";
}

// Position score marker inside its threshold-calibrated segment on the 6-band bar
export function scoreToBarPercent(score, thresholds) {
  const bounds = [0, ...thresholds, 5];
  const clamped = Math.max(0, Math.min(5, score));
  const idx = applyThresholds(clamped, thresholds);
  const lo = bounds[idx];
  const hi = bounds[idx + 1];
  const frac = hi > lo ? Math.max(0, Math.min(1, (clamped - lo) / (hi - lo))) : 0.5;
  return ((idx + frac) / 6) * 100;
}

// Shared sentence/word tokenizer (1-to-1 with train_model.py)
export function splitSentences(text) {
  const cleaned = text
    .trim()
    .replace(/’/g, "'")
    .replace(ABBREV_RE, m => m.replace(/\./g, "·"));

  return cleaned
    .split(SENT_SPLIT_RE)
    .map(s => s.trim())
    .filter(Boolean)
    .map(s => s.match(WORD_RE) || [])
    .filter(words => words.length > 0);
}

// Compute the 10-feature vector + UI stats from tokenized sentences
export function extractFeaturesFromSentences(sentWordLists, dictionary, modelMeta) {
  const rareThreshold = modelMeta.rare_zipf_threshold ?? 4.0;
  const sentLengths = sentWordLists.map(w => w.length);
  const allWords = sentWordLists.flat();

  const numWords = allWords.length;
  const numSents = sentLengths.length;
  if (numWords === 0 || numSents === 0) return null;

  const avgSentLen = numWords / numSents;
  const maxSentLen = Math.max(...sentLengths);
  const variance = sentLengths.reduce((sum, len) => sum + (len - avgSentLen) ** 2, 0) / numSents;
  const stdSentLen = Math.sqrt(variance);

  const avgWordLen = allWords.reduce((sum, w) => sum + w.length, 0) / numWords;
  const longWordRatio = allWords.filter(w => w.length >= 10).length / numWords;

  const uniqueLemmas = new Set();
  const validZipf = [];
  const nt2FirstScores = [];
  let knownDictWords = 0;
  let nt2A1A2 = 0;
  let nt2B2Plus = 0;
  let nt2Oov = 0;

  for (const w of allWords) {
    const resolved = resolveEntry(dictionary, w);
    if (resolved) {
      knownDictWords++;
      uniqueLemmas.add(resolved.lemma);
      if (resolved.z > 0) validZipf.push(resolved.z);
      if (resolved.c !== null) {
        nt2FirstScores.push(resolved.c);
        if (resolved.c <= 1) nt2A1A2++;
        else if (resolved.c >= 3) nt2B2Plus++;
      } else {
        nt2Oov++;
      }
    } else {
      uniqueLemmas.add(w.toLowerCase());
      nt2Oov++;
    }
  }

  const guiraudTtr = uniqueLemmas.size / Math.sqrt(numWords);

  let p10Zipf = 0;
  let rareWordRatio = 0;
  if (validZipf.length > 0) {
    p10Zipf = percentile(validZipf, 10);
    rareWordRatio = validZipf.filter(z => z < rareThreshold).length / validZipf.length;
  }

  const nt2MeanFirst =
    nt2FirstScores.length > 0
      ? nt2FirstScores.reduce((a, b) => a + b, 0) / nt2FirstScores.length
      : 4.0;

  return {
    features: [
      avgSentLen,
      maxSentLen,
      stdSentLen,
      avgWordLen,
      longWordRatio,
      guiraudTtr,
      p10Zipf,
      rareWordRatio,
      nt2MeanFirst,
      nt2Oov / numWords
    ],
    stats: {
      numWords,
      avgSentLen,
      a1a2Pct: (nt2A1A2 / numWords) * 100,
      b2PlusPct: (nt2B2Plus / numWords) * 100,
      dictCoverage: knownDictWords / numWords
    }
  };
}

// Group long texts (> 350 words) into ~200-word chunks matching training distribution
export function buildChunks(sentWordLists) {
  const chunks = [];
  let cur = [];
  let curWords = 0;

  for (const words of sentWordLists) {
    cur.push(words);
    curWords += words.length;
    if (curWords >= CHUNK_TARGET_WORDS) {
      chunks.push({ sents: cur, words: curWords });
      cur = [];
      curWords = 0;
    }
  }
  if (cur.length > 0) {
    if (chunks.length > 0 && curWords < 60) {
      chunks[chunks.length - 1].sents.push(...cur);
      chunks[chunks.length - 1].words += curWords;
    } else {
      chunks.push({ sents: cur, words: curWords });
    }
  }
  return chunks;
}

// Classify a token for UI vocabulary highlighting (B1+, rare non-NT2, or unlisted compound)
export function classifyVocabToken(segment, dictionary, modelMeta) {
  const rareThreshold = modelMeta.rare_zipf_threshold ?? 4.0;
  const nt2Labels = modelMeta.nt2_levels || ["A1", "A2", "B1", "B2", "C1"];
  const resolved = resolveEntry(dictionary, segment) || resolveInflectedUi(dictionary, segment);

  if (resolved && resolved.c !== null) {
    if (resolved.c <= 1 || resolved.z >= 5.5) return null;

    const lvlLabel = nt2Labels[resolved.c];
    const cssKey = lvlLabel.toLowerCase();
    let definition = resolved.d;

    if (!definition) {
      const comp = splitCompound(dictionary, segment);
      definition = comp ? comp.d : "NT2Lex vocabulary (no English gloss available)";
    }

    return {
      key: resolved.lemma,
      word: resolved.lemma,
      badge: lvlLabel,
      cssKey,
      sortRank: resolved.c,
      definition
    };
  }

  if (resolved && resolved.d && resolved.z > 0 && resolved.z < rareThreshold) {
    return {
      key: resolved.lemma,
      word: resolved.lemma,
      badge: `Rare · ${resolved.z}`,
      cssKey: "rare",
      sortRank: 3.5,
      definition: resolved.d
    };
  }

  if (!resolved || !resolved.d) {
    const compound = splitCompound(dictionary, segment);
    if (compound) {
      return {
        key: compound.lemma,
        word: compound.lemma,
        badge: "Compound",
        cssKey: "rare",
        sortRank: 2.5,
        definition: compound.d
      };
    }
  }

  return null;
}