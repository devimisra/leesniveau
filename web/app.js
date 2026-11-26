let session = null;
let dictionary = null;
let modelMeta = null;

const ABBREV_RE =
  /\b(?:o\.a|bijv|d\.w\.z|i\.p\.v|m\.b\.v|m\.a\.w|t\.o\.v|t\.a\.v|i\.v\.m|n\.a\.v|o\.b\.v|o\.i\.d|e\.d|enz|etc|nl|dr|drs|prof|mr|ir|dhr|mw|mevr|fam|blz|pag|nr|ca|st|max|min|tel|a\.u\.b|s\.v\.p)\./gi;
const SENT_SPLIT_RE = /[.!?]+(?:\s+|$)|\r?\n+/;
const WORD_RE = /[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*/g;
const LONG_TEXT_WORD_LIMIT = 350;
const CHUNK_TARGET_WORDS = 200;

const SAMPLES = {
  a2:
    "Het NOS Journaal in Makkelijke Taal heeft minder onderwerpen dan de andere journaals. " +
    "De presentator geeft meer uitleg en er worden geen moeilijke woorden gebruikt. " +
    "Ook het tempo en het gebruik van beeld is rustiger. " +
    "Kijkers van het eerdere weekjournaal (NOS Nieuws van de Week) vinden dat prettig, zo blijkt uit hun reacties.",
  b2:
    "Nederlandse gemeenten maken zich steeds meer zorgen over het tekort aan betaalbare huurwoningen. " +
    "Hoewel het kabinet vorig jaar nieuwe maatregelen aankondigde om de woningbouw te versnellen, " +
    "lopen veel projecten vertraging op door gestegen bouwkosten en stikstofregels. " +
    "Woningcorporaties pleiten daarom voor snellere vergunningsprocedures en meer financiële steun vanuit Den Haag.",
  c1:
    "De toenemende verwevenheid van geopolitieke belangen en mondiale toeleveringsketens dwingt Europese beleidsmakers " +
    "tot een fundamentele heroverweging van hun industriestrategie. Waar decennialang werd uitgegaan van ongehinderde " +
    "marktintegratie als waarborg voor welvaart, verschuift de nadruk thans naar strategische autonomie en weerbaarheid. " +
    "Deze koerswijziging roept echter complexe vraagstukken op omtrent mededinging, subsidies en de verdeling van maatschappelijke lasten."
};

const statusEl = document.getElementById("status");
const inputEl = document.getElementById("inputText");
const gradeBtn = document.getElementById("gradeBtn");
const clearBtn = document.getElementById("clearBtn");
const resultsEl = document.getElementById("results");
const cefrBadgeEl = document.getElementById("cefrBadge");
const scoreTextEl = document.getElementById("scoreText");
const statsGridEl = document.getElementById("statsGrid");
const scaleMarkerEl = document.getElementById("scaleMarker");
const shortWarningEl = document.getElementById("shortWarning");
const langWarningEl = document.getElementById("langWarning");
const annotatedTextEl = document.getElementById("annotatedText");
const vocabListEl = document.getElementById("vocabList");
const vocabCountEl = document.getElementById("vocabCount");

// 1. Load model_meta.json fresh, then load version-cached ONNX model & dictionary
async function init() {
  try {
    const metaResp = await fetch("./model_meta.json", { cache: "no-store" });
    if (!metaResp.ok) throw new Error(`model_meta.json HTTP ${metaResp.status}`);
    modelMeta = await metaResp.json();

    const v = encodeURIComponent(modelMeta.version || "1");
    const [modelResp, dictResp] = await Promise.all([
      fetch(`./leesniveau_model.onnx?v=${v}`),
      fetch(`./dictionary.json?v=${v}`)
    ]);

    if (!modelResp.ok) throw new Error(`leesniveau_model.onnx HTTP ${modelResp.status}`);
    if (!dictResp.ok) throw new Error(`dictionary.json HTTP ${dictResp.status}`);

    const [modelBuffer, dictJson] = await Promise.all([
      modelResp.arrayBuffer(),
      dictResp.json()
    ]);

    session = await ort.InferenceSession.create(modelBuffer);
    dictionary = dictJson;

    statusEl.textContent = `Ready (model v${modelMeta.version}). Paste Dutch text below to grade.`;
    gradeBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = `Error loading assets: ${err.message}`;
    console.error(err);
  }
}

// Safe dictionary lookup (guards against JS Object.prototype keys like "constructor")
function lookupExact(lower) {
  if (!Object.prototype.hasOwnProperty.call(dictionary, lower)) return null;
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

// Helper: resolve inflections, 's plurals (auto's -> auto), and hyphenated forms (e-mail -> email)
function resolveEntry(rawWord) {
  const lower = rawWord.toLowerCase().replace(/’/g, "'");
  let hit = lookupExact(lower);
  if (hit) return hit;

  if (lower.endsWith("'s") && lower.length > 3) {
    hit = lookupExact(lower.slice(0, -2));
    if (hit) return hit;
  }
  if (lower.includes("-")) {
    hit = lookupExact(lower.replace(/-/g, ""));
    if (hit) return hit;
  }
  return null;
}

// Helper: split unlisted Dutch compounds (e.g., weekjournaal -> week + journaal)
function splitCompound(rawWord) {
  if (rawWord.length < 6 || /^[A-Z0-9]+$/.test(rawWord)) return null;

  const lower = rawWord.toLowerCase();
  const linkers = ["", "s", "e", "en"];
  let best = null;

  for (let i = 3; i <= lower.length - 3; i++) {
    const leftRes = resolveEntry(lower.slice(0, i));
    if (!leftRes || !leftRes.d || leftRes.z < 2.5) continue;

    for (const linker of linkers) {
      if (linker && !lower.startsWith(linker, i)) continue;
      const rightRaw = lower.slice(i + linker.length);
      if (rightRaw.length < 3) continue;

      const rightRes = resolveEntry(rightRaw);
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
function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// Matches np.digitize(score, thresholds)
function applyThresholds(score, thresholds) {
  let idx = 0;
  while (idx < thresholds.length && score >= thresholds[idx]) {
    idx++;
  }
  return idx;
}

// Position score marker inside its threshold-calibrated segment on the 6-band bar
function scoreToBarPercent(score, thresholds) {
  const bounds = [0, ...thresholds, 5];
  const clamped = Math.max(0, Math.min(5, score));
  const idx = applyThresholds(clamped, thresholds);
  const lo = bounds[idx];
  const hi = bounds[idx + 1];
  const frac = hi > lo ? Math.max(0, Math.min(1, (clamped - lo) / (hi - lo))) : 0.5;
  return ((idx + frac) / 6) * 100;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// 2. Shared sentence/word tokenizer (1-to-1 with train_model.py)
function splitSentences(text) {
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
function extractFeaturesFromSentences(sentWordLists) {
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
    const resolved = resolveEntry(w);
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
function buildChunks(sentWordLists) {
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

// 3. Run ONNX regression inference
async function predictCEFR(features, logParity = true) {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const tensor = new ort.Tensor("float32", Float32Array.from(features), [1, features.length]);
  const results = await session.run({ [inputName]: tensor });

  const rawScore = Number(results[outputName].data[0]);
  const labelIndex = applyThresholds(rawScore, modelMeta.thresholds);
  const label = modelMeta.labels[labelIndex] || "Unknown";

  if (logParity) {
    const debugTable = Object.fromEntries(
      modelMeta.features.map((name, i) => [name, Number(features[i].toFixed(4))])
    );
    console.log("JS Parity Check:", debugTable, { rawScore: Number(rawScore.toFixed(4)), label });
  }

  return { label, rawScore };
}

// Classify a token for UI vocabulary highlighting (B1+, rare non-NT2, or unlisted compound)
function classifyVocabToken(segment) {
  const rareThreshold = modelMeta.rare_zipf_threshold ?? 4.0;
  const nt2Labels = modelMeta.nt2_levels || ["A1", "A2", "B1", "B2", "C1"];
  const resolved = resolveEntry(segment);

  if (resolved && resolved.c !== null) {
    // Skip A1/A2 words and top-frequency homographs (Zipf >= 5.5)
    if (resolved.c <= 1 || resolved.z >= 5.5) return null;

    const lvlLabel = nt2Labels[resolved.c];
    const cssKey = lvlLabel.toLowerCase();
    let definition = resolved.d;

    if (!definition) {
      const comp = splitCompound(segment);
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
    const compound = splitCompound(segment);
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

// 4. Render highlighted text and sorted study vocabulary list
function renderVocabulary(text) {
  const wordSegmenter = new Intl.Segmenter("nl", { granularity: "word" });
  const vocabMap = new Map();
  let html = "";

  for (const { segment, isWordLike } of wordSegmenter.segment(text)) {
    if (!isWordLike) {
      html += escapeHtml(segment);
      continue;
    }

    const item = classifyVocabToken(segment);
    if (item) {
      const tooltip = escapeHtml(`[${item.badge}] ${item.word}: ${item.definition}`);
      html += `<span class="hl hl-${item.cssKey}" title="${tooltip}">${escapeHtml(segment)}</span>`;
      if (!vocabMap.has(item.key)) {
        vocabMap.set(item.key, item);
      }
    } else {
      html += escapeHtml(segment);
    }
  }

  annotatedTextEl.innerHTML = html;

  const sortedItems = [...vocabMap.values()].sort(
    (a, b) => b.sortRank - a.sortRank || a.word.localeCompare(b.word, "nl")
  );
  vocabCountEl.textContent = String(sortedItems.length);
  vocabListEl.innerHTML = "";

  if (sortedItems.length === 0) {
    const li = document.createElement("li");
    li.textContent = "All recognized vocabulary is A1–A2 level.";
    vocabListEl.appendChild(li);
    return;
  }

  for (const item of sortedItems) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="badge badge-${item.cssKey}">${escapeHtml(item.badge)}</span> ` +
      `<strong>${escapeHtml(item.word)}</strong>: ${escapeHtml(item.definition)}`;
    vocabListEl.appendChild(li);
  }
}

function renderStats(stats) {
  statsGridEl.innerHTML = `
    <div class="stat-item">
      <span class="stat-val">${stats.numWords}</span>
      <span class="stat-key">Words</span>
    </div>
    <div class="stat-item">
      <span class="stat-val">${stats.avgSentLen.toFixed(1)}</span>
      <span class="stat-key">Avg Sent Len</span>
    </div>
    <div class="stat-item">
      <span class="stat-val">${stats.a1a2Pct.toFixed(0)}%</span>
      <span class="stat-key">A1–A2 Vocab</span>
    </div>
    <div class="stat-item">
      <span class="stat-val">${stats.b2PlusPct.toFixed(0)}%</span>
      <span class="stat-key">B2+ Vocab</span>
    </div>
  `;
  shortWarningEl.hidden = stats.numWords >= 30;
}

async function runGrade() {
  try {
    const text = inputEl.value.trim();
    if (!text) {
      resultsEl.hidden = true;
      statusEl.textContent = "Please paste some Dutch text first.";
      return;
    }

    const sentWordLists = splitSentences(text);
    const fullDoc = extractFeaturesFromSentences(sentWordLists);
    if (!fullDoc) {
      resultsEl.hidden = true;
      statusEl.textContent = "No Dutch words detected. Please paste text containing letters.";
      return;
    }

    let rawScore;
    let label;
    let chunkCount = 1;

    if (fullDoc.stats.numWords > LONG_TEXT_WORD_LIMIT) {
      const chunks = buildChunks(sentWordLists);
      chunkCount = chunks.length;
      let weightedSum = 0;
      let totalWeight = 0;

      for (const ch of chunks) {
        const chExtracted = extractFeaturesFromSentences(ch.sents);
        const res = await predictCEFR(chExtracted.features, false);
        weightedSum += res.rawScore * ch.words;
        totalWeight += ch.words;
      }

      rawScore = weightedSum / totalWeight;
      const labelIdx = applyThresholds(rawScore, modelMeta.thresholds);
      label = modelMeta.labels[labelIdx] || "Unknown";
      console.log(`Chunked Inference (${chunkCount} sections):`, {
        rawScore: Number(rawScore.toFixed(4)),
        label,
        dictCoverage: Number(fullDoc.stats.dictCoverage.toFixed(3))
      });
    } else {
      const res = await predictCEFR(fullDoc.features, true);
      rawScore = res.rawScore;
      label = res.label;
    }

    cefrBadgeEl.textContent = label;
    const chunkNote = chunkCount > 1 ? ` · avg of ${chunkCount} sections` : "";
    scoreTextEl.textContent = `raw score: ${rawScore.toFixed(2)} / 5.00${chunkNote}`;

    scaleMarkerEl.style.left = `${scoreToBarPercent(rawScore, modelMeta.thresholds)}%`;
    scaleMarkerEl.hidden = false;

    renderStats(fullDoc.stats);
    langWarningEl.hidden = fullDoc.stats.numWords < 5 || fullDoc.stats.dictCoverage >= 0.55;
    renderVocabulary(text);
    resultsEl.hidden = false;
    statusEl.textContent = `Ready (model v${modelMeta.version}).`;
  } catch (err) {
    statusEl.textContent = `Prediction error: ${err.message}`;
    console.error(err);
  }
}

gradeBtn.addEventListener("click", runGrade);

clearBtn.addEventListener("click", () => {
  inputEl.value = "";
  resultsEl.hidden = true;
  shortWarningEl.hidden = true;
  langWarningEl.hidden = true;
  if (modelMeta) statusEl.textContent = `Ready (model v${modelMeta.version}). Paste Dutch text below to grade.`;
  inputEl.focus();
});

document.querySelectorAll(".sample-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sample;
    if (!SAMPLES[key]) return;
    inputEl.value = SAMPLES[key];
    if (!gradeBtn.disabled) runGrade();
  });
});

init();