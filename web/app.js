import {
  WORD_RE,
  CLAUSE_SPLIT_RE,
  LONG_TEXT_WORD_LIMIT,
  applyThresholds,
  getBorderlineNote,
  scoreToBarPercent,
  splitSentences,
  extractFeaturesFromSentences,
  buildChunks,
  findSeparableMatches,
  classifyResolvedEntry,
  classifyVocabToken
} from "./features.js?v=1.5.0";

let session = null;
let dictionary = null;
let modelMeta = null;
let lastVocabItems = [];
let pdfjsPromise = null;
let mammothPromise = null;

const PDFJS_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
const PDFJS_WORKER_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
const MAMMOTH_URL = "https://cdn.jsdelivr.net/npm/mammoth@1.11.0/mammoth.browser.min.js";
const MAX_FILE_BYTES = 15 * 1024 * 1024;

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
    "Deze koerswijziging roept echter complexe vraagstukken op omtrent mededinging, staatssteun en de rechtvaardige verdeling van maatschappelijke lasten.\n\n" +
    "Bovendien waarschuwen economen dat een ongecoördineerde wedloop om subsidies niet alleen de begrotingsdiscipline ondermijnt, " +
    "maar ook het gelijke speelveld binnen de interne markt verstoort doordat lidstaten met omvangrijke financiële buffers hun eigen " +
    "sleuteltechnologieën aanzienlijk royaler kunnen ondersteunen dan kleinere economieën. Tegelijkertijd vergt de verduurzaming van " +
    "energie-intensieve sectoren langjarige investeringszekerheid, heldere vergunningskaders en hoogwaardige kennisinfrastructuur. " +
    "Zonder een samenhangende Europese aanpak dreigt het continent immers zowel zijn concurrentievermogen als zijn technologische " +
    "slagkracht op de langere termijn te verliezen.\n\n" +
    "Het kabinet pleit daarom voor gerichte publiek-private samenwerking, waarbij schaarse overheidsmiddelen uitsluitend worden ingezet " +
    "voor hoogwaardige innovaties die zonder publieke cofinanciering onvoldoende van de grond komen en waarvan de maatschappelijke baten " +
    "breder neerslaan dan bij individuele ondernemingen alleen."
};

const statusEl = document.getElementById("status");
const inputEl = document.getElementById("inputText");
const gradeBtn = document.getElementById("gradeBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInputEl = document.getElementById("fileInput");
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
const exportAnkiBtn = document.getElementById("exportAnkiBtn");
const exportCsvBtn = document.getElementById("exportCsvBtn");

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

    statusEl.textContent = `Ready (model v${modelMeta.version}). Paste Dutch text or drop a file below.`;
    gradeBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = `Error loading assets: ${err.message}`;
    console.error(err);
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

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

function renderVocabulary(text) {
  const sentSegmenter = new Intl.Segmenter("nl", { granularity: "sentence" });
  const wordSegmenter = new Intl.Segmenter("nl", { granularity: "word" });
  const vocabMap = new Map();
  let html = "";

  for (const { segment: sentence } of sentSegmenter.segment(text)) {
    const cleanContext = sentence.replace(/\s+/g, " ").trim();

    // Pre-detect separable verb pairs per clause inside this sentence
    const sepOverrideQueue = [];
    for (const clauseStr of sentence.split(CLAUSE_SPLIT_RE)) {
      const clauseWords = clauseStr.match(WORD_RE) || [];
      const sepMatches = findSeparableMatches(clauseWords, dictionary);
      for (let i = 0; i < clauseWords.length; i++) {
        sepOverrideQueue.push({
          surface: clauseWords[i],
          sep: sepMatches.get(i) || null
        });
      }
    }
    let qIdx = 0;

    for (const { segment, isWordLike } of wordSegmenter.segment(sentence)) {
      if (!isWordLike) {
        html += escapeHtml(segment);
        continue;
      }

      let sepOverride = null;
      while (qIdx < sepOverrideQueue.length) {
        const entry = sepOverrideQueue[qIdx++];
        if (entry.surface === segment) {
          sepOverride = entry.sep;
          break;
        }
      }

      const item = sepOverride
        ? classifyResolvedEntry(sepOverride.resolved, sepOverride.resolved.lemma, dictionary, modelMeta)
        : classifyVocabToken(segment, dictionary, modelMeta);

      if (item) {
        const tooltip = escapeHtml(`[${item.badge}] ${item.word}: ${item.definition}`);
        html += `<span class="hl hl-${item.cssKey}" title="${tooltip}">${escapeHtml(segment)}</span>`;
        if (!vocabMap.has(item.key)) {
          vocabMap.set(item.key, { ...item, context: cleanContext });
        }
      } else {
        html += escapeHtml(segment);
      }
    }
  }
  
  annotatedTextEl.innerHTML = html;

  const sortedItems = [...vocabMap.values()].sort(
    (a, b) => b.sortRank - a.sortRank || a.word.localeCompare(b.word, "nl")
  );
  lastVocabItems = sortedItems;
  vocabCountEl.textContent = String(sortedItems.length);
  exportAnkiBtn.disabled = sortedItems.length === 0;
  exportCsvBtn.disabled = sortedItems.length === 0;
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

function triggerDownload(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function cleanTsvField(str) {
  return String(str || "").replace(/[\t\r\n]+/g, " ").trim();
}

function escapeCsvField(str) {
  const s = String(str || "").replace(/\r?\n/g, " ");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportAnkiTsv() {
  if (!lastVocabItems.length) return;
  const header = [
    "#separator:tab",
    "#html:false",
    "#columns:Word\tDefinition\tLevel\tContext\tTags"
  ];
  const rows = lastVocabItems.map(item => {
    const tag = `leesniveau::${item.badge.split(" ")[0].toLowerCase()}`;
    return [
      cleanTsvField(item.word),
      cleanTsvField(item.definition),
      cleanTsvField(item.badge),
      cleanTsvField(item.context),
      tag
    ].join("\t");
  });
  const stamp = new Date().toISOString().slice(0, 10);
  triggerDownload(
    `leesniveau-anki-${stamp}.tsv`,
    [...header, ...rows].join("\n") + "\n",
    "text/tab-separated-values;charset=utf-8"
  );
}

function exportCsv() {
  if (!lastVocabItems.length) return;
  const header = "Word,Level,Definition,Context";
  const rows = lastVocabItems.map(item =>
    [item.word, item.badge, item.definition, item.context].map(escapeCsvField).join(",")
  );
  const stamp = new Date().toISOString().slice(0, 10);
  triggerDownload(
    `leesniveau-vocab-${stamp}.csv`,
    "\uFEFF" + [header, ...rows].join("\r\n") + "\r\n",
    "text/csv;charset=utf-8"
  );
}

function stripMarkdown(md) {
  return md
    .replace(/```[\s\S]*?```/g, "\n")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^>\s?/gm, "")
    .replace(/[*_~`]+/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function stripSubtitles(raw) {
  return raw
    .replace(/^\uFEFF/, "")
    .replace(/^WEBVTT[^\n]*\n+/i, "")
    .replace(/^(?:NOTE|STYLE)[\s\S]*?(?:\r?\n\r?\n|$)/gim, "")
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line && !/^\d+$/.test(line) && !line.includes("-->"))
    .map(line => line.replace(/<[^>]+>/g, "").trim())
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

async function getPdfJs() {
  if (!pdfjsPromise) {
    pdfjsPromise = import(PDFJS_URL).then(lib => {
      lib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
      return lib;
    });
  }
  return pdfjsPromise;
}

async function extractPdfText(arrayBuffer) {
  const pdfjsLib = await getPdfJs();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageTexts = [];

  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    let out = "";
    let prevY = null;
    let prevHeight = 12;

    for (const item of content.items) {
      const str = (item.str || "").trim();
      if (!str) continue;

      const y = Array.isArray(item.transform) ? item.transform[5] : prevY;
      const height = item.height > 0 ? item.height : prevHeight;

      if (prevY !== null && Math.abs(y - prevY) > height * 1.6) {
        out += "\n\n" + str;
      } else if (/[a-zà-ÿ]-$/i.test(out) && /^[a-zà-ÿ]/.test(str)) {
        out = out.slice(0, -1) + str;
      } else if (out && !out.endsWith("\n")) {
        out += " " + str;
      } else {
        out += str;
      }

      prevY = y;
      prevHeight = height;
    }

    if (out.trim()) pageTexts.push(out.trim());
  }

  const combined = pageTexts.join("\n\n").replace(/[ \t]{2,}/g, " ").trim();
  if (!WORD_RE.test(combined)) {
    throw new Error("No selectable text found in this PDF (it may be a scanned image).");
  }
  return combined;
}

function getMammoth() {
  if (window.mammoth) return Promise.resolve(window.mammoth);
  if (!mammothPromise) {
    mammothPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = MAMMOTH_URL;
      script.onload = () =>
        window.mammoth ? resolve(window.mammoth) : reject(new Error("Failed to initialize mammoth."));
      script.onerror = () => reject(new Error("Could not load .docx parser from CDN."));
      document.head.appendChild(script);
    });
  }
  return mammothPromise;
}

async function extractDocxText(arrayBuffer) {
  const mammoth = await getMammoth();
  const result = await mammoth.extractRawText({ arrayBuffer });
  return (result.value || "").trim();
}

async function handleFile(file) {
  if (!file) return;
  if (file.size > MAX_FILE_BYTES) {
    statusEl.textContent = "File is too large (limit: 15 MB).";
    return;
  }

  const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
  statusEl.textContent = `Reading ${file.name} locally in your browser...`;

  try {
    let extracted = "";
    if (ext === "txt") {
      extracted = await file.text();
    } else if (ext === "md" || ext === "markdown") {
      extracted = stripMarkdown(await file.text());
    } else if (ext === "srt" || ext === "vtt") {
      extracted = stripSubtitles(await file.text());
    } else if (ext === "pdf") {
      extracted = await extractPdfText(await file.arrayBuffer());
    } else if (ext === "docx") {
      extracted = await extractDocxText(await file.arrayBuffer());
    } else {
      throw new Error("Unsupported file type. Use .txt, .md, .pdf, .docx, .srt, or .vtt.");
    }

    inputEl.value = extracted.trim();
    if (!gradeBtn.disabled) {
      await runGrade();
      statusEl.textContent = `Loaded ${file.name} locally (model v${modelMeta.version}).`;
    } else {
      statusEl.textContent = `Loaded ${file.name}. Waiting for model to finish loading...`;
    }
  } catch (err) {
    statusEl.textContent = `File error: ${err.message}`;
    console.error(err);
  } finally {
    fileInputEl.value = "";
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
    const fullDoc = extractFeaturesFromSentences(sentWordLists, dictionary, modelMeta);
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
        const chExtracted = extractFeaturesFromSentences(ch.sents, dictionary, modelMeta);
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
    const borderlineNote = getBorderlineNote(rawScore, modelMeta.thresholds, modelMeta.labels);
    scoreTextEl.textContent = `raw score: ${rawScore.toFixed(2)} / 5.00${chunkNote}${borderlineNote}`;

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
exportAnkiBtn.addEventListener("click", exportAnkiTsv);
exportCsvBtn.addEventListener("click", exportCsv);

uploadBtn.addEventListener("click", () => fileInputEl.click());
fileInputEl.addEventListener("change", e => handleFile(e.target.files[0]));

inputEl.addEventListener("dragover", e => {
  e.preventDefault();
  inputEl.classList.add("drag-over");
});
inputEl.addEventListener("dragleave", () => inputEl.classList.remove("drag-over"));
inputEl.addEventListener("drop", e => {
  e.preventDefault();
  inputEl.classList.remove("drag-over");
  if (e.dataTransfer?.files?.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

clearBtn.addEventListener("click", () => {
  inputEl.value = "";
  lastVocabItems = [];
  exportAnkiBtn.disabled = true;
  exportCsvBtn.disabled = true;
  resultsEl.hidden = true;
  shortWarningEl.hidden = true;
  langWarningEl.hidden = true;
  if (modelMeta) statusEl.textContent = `Ready (model v${modelMeta.version}). Paste Dutch text or drop a file below.`;
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