export const ABBREV_RE =
  /\b(?:o\.a|bijv|d\.w\.z|i\.p\.v|m\.b\.v|m\.a\.w|t\.o\.v|t\.a\.v|i\.v\.m|n\.a\.v|o\.b\.v|o\.i\.d|e\.d|enz|etc|nl|dr|drs|prof|mr|ir|dhr|mw|mevr|fam|blz|pag|nr|ca|st|max|min|tel|a\.u\.b|s\.v\.p)\./gi;
export const SENT_SPLIT_RE = /[.!?]+(?:\s+|$)|\r?\n+/;
export const CLAUSE_SPLIT_RE =
  /[,;:()–—]+|\b(?:en|maar|want|of|dus|omdat|dat|toen|terwijl|hoewel|indien|zodat|waardoor|waarbij|als|wanneer|omtrent)\b/gi;
export const WORD_RE = /[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*/g;
export const LONG_TEXT_WORD_LIMIT = 350;
export const CHUNK_TARGET_WORDS = 200;

export const SEPARABLE_PARTICLES = new Set([
  "aan", "achter", "af", "bij", "binnen", "boven", "buiten", "door", "gelijk", "goed",
  "in", "langs", "mee", "mis", "na", "neer", "om", "onder", "op", "open", "over",
  "rond", "samen", "schoon", "stil", "tegen", "terug", "thuis", "toe", "tussen",
  "uit", "vast", "voor", "voort", "vrij", "weg", "weer"
]);

const AUX_OR_STOP_VERBS = new Set([
  "zijn", "hebben", "worden", "kunnen", "moeten", "zullen", "willen", "mogen",
  "de", "het", "een", "dat", "die", "wat", "wie", "waar", "hoe"
]);

function isVerbInfinitive(lemma) {
  return /(?:en|aan|oen|ijn|ien)$/.test(lemma) && lemma.length >= 5;
}

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

// Match clause-final separable particle with earlier finite verb in the same clause
// Returns Map<tokenIndex, { resolved, surfaceLength, role: "verb" | "particle" }>
export function findSeparableMatches(clauseWords, dictionary) {
  const matches = new Map();
  if (clauseWords.length < 2) return matches;

  const lastIdx = clauseWords.length - 1;
  const particleLower = clauseWords[lastIdx].toLowerCase();
  if (!SEPARABLE_PARTICLES.has(particleLower)) return matches;

  for (let i = 0; i < lastIdx; i++) {
    const rawVerb = clauseWords[i];
    const verbLower = rawVerb.toLowerCase();
    if (SEPARABLE_PARTICLES.has(verbLower)) continue;

    const verbResolved = resolveEntry(dictionary, rawVerb);
    const stemLemmas = [];
    if (verbResolved && isVerbInfinitive(verbResolved.lemma)) {
      stemLemmas.push(verbResolved.lemma);
    }
    // Lightweight fallback for finite verb stems not explicitly linked in Wiktionary
    for (const suf of ["ten", "den", "te", "de", "t"]) {
      if (verbLower.endsWith(suf) && verbLower.length - suf.length >= 3) {
        const root = verbLower.slice(0, -suf.length);
        stemLemmas.push(root + "en");
        if (root.length >= 2 && root.at(-1) === root.at(-2)) {
          stemLemmas.push(root.slice(0, -1) + "en");
        }
      }
    }

    for (const stem of stemLemmas) {
      if (AUX_OR_STOP_VERBS.has(stem)) continue;
      const candidate = particleLower + stem;
      const compHit = lookupExact(dictionary, candidate);
      if (compHit && (compHit.d || compHit.c !== null)) {
        const mergedLen = rawVerb.length + clauseWords[lastIdx].length;
        matches.set(i, { resolved: compHit, surfaceLength: mergedLen, role: "verb" });
        matches.set(lastIdx, { resolved: compHit, surfaceLength: mergedLen, role: "particle" });
        return matches;
      }
    }
  }

  return matches;
}

// Split raw sentence string into word arrays per clause (shared by feature extraction & UI)
export function splitSentenceClauses(sentenceStr) {
  return sentenceStr
    .split(CLAUSE_SPLIT_RE)
    .map(c => c.match(WORD_RE) || [])
    .filter(words => words.length > 0);
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

// Shared sentence/clause tokenizer (1-to-1 with train_model.py)
// Returns array of sentences, where each sentence is an array of clause word arrays
export function splitSentences(text) {
  const cleaned = text
    .trim()
    .replace(/’/g, "'")
    .replace(ABBREV_RE, m => m.replace(/\./g, "·"));

  return cleaned
    .split(SENT_SPLIT_RE)
    .map(s => s.trim())
    .filter(Boolean)
    .map(s => splitSentenceClauses(s))
    .filter(clauses => clauses.length > 0);
}

// Compute the 11-feature vector + UI stats from tokenized sentence clauses
export function extractFeaturesFromSentences(sentClauseLists, dictionary, modelMeta) {
  const rareThreshold = modelMeta.rare_zipf_threshold ?? 4.0;
  const sentLengths = [];
  const mergedTokens = []; // { wordLen, resolved, rawLower }

  for (const clauses of sentClauseLists) {
    let sentWordCount = 0;
    for (const clauseWords of clauses) {
      const sepMatches = findSeparableMatches(clauseWords, dictionary);
      for (let i = 0; i < clauseWords.length; i++) {
        const sep = sepMatches.get(i);
        if (sep) {
          if (sep.role === "particle") continue;
          sentWordCount++;
          mergedTokens.push({
            wordLen: sep.surfaceLength,
            resolved: sep.resolved,
            rawLower: sep.resolved.lemma
          });
        } else {
          const w = clauseWords[i];
          sentWordCount++;
          mergedTokens.push({
            wordLen: w.length,
            resolved: resolveEntry(dictionary, w),
            rawLower: w.toLowerCase()
          });
        }
      }
    }
    if (sentWordCount > 0) sentLengths.push(sentWordCount);
  }

  const numWords = mergedTokens.length;
  const numSents = sentLengths.length;
  if (numWords === 0 || numSents === 0) return null;

  const avgSentLen = numWords / numSents;
  const maxSentLen = Math.max(...sentLengths);
  const variance = sentLengths.reduce((sum, len) => sum + (len - avgSentLen) ** 2, 0) / numSents;
  const stdSentLen = Math.sqrt(variance);

  const avgWordLen = mergedTokens.reduce((sum, t) => sum + t.wordLen, 0) / numWords;
  const longWordRatio = mergedTokens.filter(t => t.wordLen >= 10).length / numWords;

  const uniqueLemmas = new Set();
  const validZipf = [];
  const nt2FirstScores = [];
  let knownDictWords = 0;
  let nt2A1A2 = 0;
  let nt2B2Plus = 0;
  let rareNt2Oov = 0;
  let commonNt2Oov = 0;

  for (const tok of mergedTokens) {
    const resolved = tok.resolved;
    if (resolved) {
      knownDictWords++;
      uniqueLemmas.add(resolved.lemma);
      if (resolved.z > 0) validZipf.push(resolved.z);
      if (resolved.c !== null) {
        nt2FirstScores.push(resolved.c);
        if (resolved.c <= 1) nt2A1A2++;
        else if (resolved.c >= 3) nt2B2Plus++;
      } else if (resolved.z > 0 && resolved.z >= rareThreshold) {
        commonNt2Oov++;
      } else {
        rareNt2Oov++;
      }
    } else {
      uniqueLemmas.add(tok.rawLower);
      rareNt2Oov++;
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
      rareNt2Oov / numWords,
      commonNt2Oov / numWords
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
export function buildChunks(sentClauseLists) {
  const chunks = [];
  let cur = [];
  let curWords = 0;

  for (const clauses of sentClauseLists) {
    const sentWords = clauses.reduce((sum, c) => sum + c.length, 0);
    cur.push(clauses);
    curWords += sentWords;
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

// Classify a resolved entry or surface token for UI vocabulary highlighting
export function classifyResolvedEntry(resolved, rawSegment, dictionary, modelMeta) {
  const rareThreshold = modelMeta.rare_zipf_threshold ?? 4.0;
  const nt2Labels = modelMeta.nt2_levels || ["A1", "A2", "B1", "B2", "C1"];

  if (resolved && resolved.c !== null) {
    if (resolved.c <= 1 || resolved.z >= 5.5) return null;

    const lvlLabel = nt2Labels[resolved.c];
    const cssKey = lvlLabel.toLowerCase();
    let definition = resolved.d;

    if (!definition) {
      const comp = splitCompound(dictionary, rawSegment);
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
    const compound = splitCompound(dictionary, rawSegment);
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

export function classifyVocabToken(segment, dictionary, modelMeta) {
  const resolved = resolveEntry(dictionary, segment) || resolveInflectedUi(dictionary, segment);
  return classifyResolvedEntry(resolved, segment, dictionary, modelMeta);
}