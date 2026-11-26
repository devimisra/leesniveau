import ast
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
SCRIPTS_DIR = ROOT / "scripts"

TEST_TEXTS = {
    "a2_sample": (
        "Het NOS Journaal in Makkelijke Taal heeft minder onderwerpen dan de andere journaals. "
        "De presentator geeft meer uitleg en er worden geen moeilijke woorden gebruikt. "
        "Ook het tempo en het gebruik van beeld is rustiger. "
        "Kijkers van het eerdere weekjournaal (NOS Nieuws van de Week) vinden dat prettig, zo blijkt uit hun reacties."
    ),
    "b2_sample": (
        "Nederlandse gemeenten maken zich steeds meer zorgen over het tekort aan betaalbare huurwoningen. "
        "Hoewel het kabinet vorig jaar nieuwe maatregelen aankondigde om de woningbouw te versnellen, "
        "lopen veel projecten vertraging op door gestegen bouwkosten en stikstofregels. "
        "Woningcorporaties pleiten daarom voor snellere vergunningsprocedures en meer financiële steun vanuit Den Haag."
    ),
    "c1_sample": (
        "De toenemende verwevenheid van geopolitieke belangen en mondiale toeleveringsketens dwingt Europese beleidsmakers "
        "tot een fundamentele heroverweging van hun industriestrategie. Waar decennialang werd uitgegaan van ongehinderde "
        "marktintegratie als waarborg voor welvaart, verschuift de nadruk thans naar strategische autonomie en weerbaarheid."
    ),
    "abbreviations_and_plurals": (
        "Prof. dr. Van Dijk besprak o.a. de nieuwe auto's, foto's en e-mails van het ministerie, "
        "d.w.z. alle documenten m.b.t. duurzame energie-efficiëntie in Nederland."
    ),
    "separable_verbs": (
        "De minister stelde het moeilijke besluit gisteren uit, "
        "maar de oppositie riep het kabinet direct ter verantwoording op."
    ),
    "prototype_guard": "De constructor van het gebouw is nog niet bekend.",
}


def load_train_module(dictionary: dict):
    """Safely load constants & feature functions from scripts/train_model.py without training imports."""
    train_path = SCRIPTS_DIR / "train_model.py"
    source = train_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(train_path))
    safe_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            safe_nodes.append(node)
        elif isinstance(node, ast.Assign):
            if all(isinstance(t, ast.Name) and t.id.isupper() and t.id != "X" for t in node.targets):
                safe_nodes.append(node)
    safe_mod = ast.Module(body=safe_nodes, type_ignores=[])
    ns = {
        "json": json,
        "math": __import__("math"),
        "re": __import__("re"),
        "np": np,
        "dictionary": dictionary,
    }
    exec(compile(safe_mod, str(train_path), "exec"), ns)
    ns["dictionary"] = dictionary
    return ns


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for JS parity check")
def test_python_js_feature_and_onnx_parity(tmp_path):
    shutil.copy2(WEB_DIR / "features.js", tmp_path / "features.mjs")
    dict_path = WEB_DIR / "dictionary.json"
    meta_path = WEB_DIR / "model_meta.json"
    onnx_path = WEB_DIR / "leesniveau_model.onnx"

    dictionary = json.loads(dict_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_ns = load_train_module(dictionary)
    py_extract = train_ns["extract_features"]

    node_script = """
    import fs from "node:fs";
    import {
      splitSentences,
      extractFeaturesFromSentences,
      findSeparableMatches,
      classifyVocabToken
    } from "./features.mjs";

    const payload = JSON.parse(fs.readFileSync(0, "utf8"));
    const dictionary = JSON.parse(fs.readFileSync(payload.dictPath, "utf8"));
    const meta = JSON.parse(fs.readFileSync(payload.metaPath, "utf8"));

    const featuresOut = {};
    for (const [key, text] of Object.entries(payload.texts)) {
      const sents = splitSentences(text);
      featuresOut[key] = extractFeaturesFromSentences(sents, dictionary, meta);
    }

    const sepClause = ["De", "minister", "stelde", "het", "besluit", "uit"];
    const sepMap = findSeparableMatches(sepClause, dictionary);

    const uiChecks = {
      snellere: classifyVocabToken("snellere", dictionary, meta),
      langjarige: classifyVocabToken("langjarige", dictionary, meta),
      heroverweging: classifyVocabToken("heroverweging", dictionary, meta),
      industriestrategie: classifyVocabToken("industriestrategie", dictionary, meta),
      separableLemma: sepMap.get(2)?.resolved?.lemma ?? null,
    };

    console.log(JSON.stringify({ featuresOut, uiChecks }));
    """

    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        input=json.dumps({
            "dictPath": str(dict_path),
            "metaPath": str(meta_path),
            "texts": TEST_TEXTS,
        }),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=True,
    )
    js_result = json.loads(proc.stdout)

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    for key, text in TEST_TEXTS.items():
        py_feats = py_extract(text)
        js_feats = js_result["featuresOut"][key]["features"]

        assert len(py_feats) == len(meta["features"]) == 11
        np.testing.assert_allclose(
            js_feats,
            py_feats,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Feature mismatch between Python and JS for '{key}'",
        )

        py_score = float(sess.run(None, {input_name: np.array([py_feats], dtype=np.float32)})[0].item())
        js_score = float(sess.run(None, {input_name: np.array([js_feats], dtype=np.float32)})[0].item())
        assert abs(py_score - js_score) < 1e-5

    # Verify UI morphological, compound, and separable-verb guards stay intact
    ui = js_result["uiChecks"]
    assert ui["snellere"] is None  # resolves to A1/A2 'snel', not 'snel + leren'
    assert ui["langjarige"] is None or ui["langjarige"]["badge"] != "Compound"
    assert ui["heroverweging"]["definition"].startswith("her- (re-)")
    assert ui["industriestrategie"]["badge"] == "Compound"
    assert ui["separableLemma"] == "uitstellen"