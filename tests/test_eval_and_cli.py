import json

from raglens.cli import main
from raglens.evaluate import QAItem, compare, evaluate, load_qa


def test_eval_metrics_on_examples(retriever, qa_path):
    items = load_qa(qa_path)
    assert len(items) >= 12
    reports = {r.mode: r for r in compare(retriever, items, k=5)}
    assert set(reports) == {"bm25", "dense", "hybrid"}
    for r in reports.values():
        assert 0 <= r.mrr <= r.hit_rate <= 1
    assert reports["hybrid"].hit_rate >= 0.9


def test_eval_counts_misses(retriever):
    r = evaluate(retriever, [QAItem("zebra quantum lasagne", ["nowhere.md"])], k=3, mode="bm25")
    assert r.hit_rate == 0 and r.mrr == 0 and r.misses == ["zebra quantum lasagne"]


def test_cli_index_ask_eval(tmp_path, docs_dir, qa_path, capsys):
    idx = tmp_path / "index.json"
    assert main(["index", str(docs_dir), "--out", str(idx)]) == 0
    assert main(["ask", "What does SEV1 mean?", "--index", str(idx), "--json"]) == 0
    out = json.loads(capsys.readouterr().out.split("\n", 2)[-1])
    assert out["citations"][0]["doc"] == "incidents.md"
    assert main(["eval", str(qa_path), "--index", str(idx), "--min-hit", "0.9"]) == 0
    assert main(["eval", str(qa_path), "--docs", str(docs_dir), "--min-hit", "1.01"]) == 2


def test_cli_search_json(docs_dir, capsys):
    assert main(["search", "multipart upload", "--docs", str(docs_dir), "--json", "-k", "2"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["id"].startswith("api-limits.md")
