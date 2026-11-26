import ast
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
SCRIPTS_DIR = ROOT / "scripts"


def load_assets():
    with open(WEB_DIR / "dictionary.json", "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    with open(WEB_DIR / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    session = ort.InferenceSession(str(WEB_DIR / "leesniveau_model.onnx"))
    return dictionary, meta, session


def load_train_ns(dictionary):
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




def test_dictionary_invariants():
    dictionary, _, _ = load_assets()
    assert len(dictionary) > 120_000
    for word, entry in dictionary.items():
        if "l" in entry:
            target = dictionary.get(entry["l"])
            assert target is not None and "d" in target and "z" in target, f"Broken pointer: {word}"
        elif "d" in entry:
            assert isinstance(entry["z"], (int, float))
        else:
            assert entry.get("c") in {0, 1, 2, 3, 4}, f"Invalid NT2-only entry: {word}"


def test_homograph_and_multisense_levels():
    dictionary, _, _ = load_assets()
    ns = load_train_ns(dictionary)
    resolve_entry = ns["resolve_entry"]

    for word in ("meer", "was", "kan", "wil"):
        res = resolve_entry(word)
        assert res is not None and res[2] == 0, f"Expected A1 (0) for {word}, got {res}"

    assert ";" in dictionary["kabinet"]["d"]
    assert ";" in dictionary["kosten"]["d"]


def test_model_meta_and_onnx_inference():
    dictionary, meta, session = load_assets()
    ns = load_train_ns(dictionary)
    extract_features = ns["extract_features"]

    assert len(meta["features"]) == 11
    assert "rare_nt2_oov_ratio" in meta["features"]
    assert "common_nt2_oov_ratio" in meta["features"]
    assert meta["thresholds"] == sorted(meta["thresholds"])
    assert len(meta["thresholds"]) == 5

    nos_text = (
        "Het NOS Journaal in Makkelijke Taal heeft minder onderwerpen dan de andere journaals. "
        "De presentator geeft meer uitleg en er worden geen moeilijke woorden gebruikt. "
        "Ook het tempo en het gebruik van beeld is rustiger. "
        "Kijkers van het eerdere weekjournaal (NOS Nieuws van de Week) vinden dat prettig, zo blijkt uit hun reacties."
    )
    feats = np.array([extract_features(nos_text)], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    raw_score = float(session.run(None, {input_name: feats})[0].item())
    pred_idx = int(np.digitize([raw_score], meta["thresholds"])[0])

    assert meta["labels"][pred_idx] == "A2"
    assert 0.8 <= raw_score <= 1.5