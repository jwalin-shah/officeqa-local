import json
import subprocess
import sys
from pathlib import Path


def test_analyze_retrieval_run_buckets(tmp_path: Path) -> None:
    details = tmp_path / "details.jsonl"
    rows = [
        {"uid": "U1", "first_hit_rank": 3},
        {"uid": "U2", "first_hit_rank": 8},
        {"uid": "U3", "first_hit_rank": 17},
        {"uid": "U4", "first_hit_rank": None},
    ]
    details.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out = tmp_path / "summary.json"
    subprocess.run(
        [sys.executable, "analyze_retrieval_run.py", str(details), "--out", str(out)],
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    summary = json.loads(out.read_text())
    assert summary["total_questions"] == 4
    assert summary["rank_buckets"]["top_5"] == 1
    assert summary["rank_buckets"]["rank_6_10"] == 1
    assert summary["rank_buckets"]["rank_11_20"] == 1
    assert summary["rank_buckets"]["miss"] == 1
    assert summary["headroom"]["move_11_20_into_top10"] == 1


def test_compare_eval_runs_reports_delta(tmp_path: Path) -> None:
    base = tmp_path / "base" / "retrieval"
    head = tmp_path / "head" / "retrieval"
    base.mkdir(parents=True)
    head.mkdir(parents=True)

    (base / "summary.json").write_text(
        json.dumps(
            {
                "recall_at_5": 10.0,
                "recall_at_10": 20.0,
                "recall_at_20": 30.0,
                "recall_at_30": 40.0,
                "recall_at_50": 50.0,
            }
        )
    )
    (head / "summary.json").write_text(
        json.dumps(
            {
                "recall_at_5": 11.0,
                "recall_at_10": 25.0,
                "recall_at_20": 35.0,
                "recall_at_30": 42.0,
                "recall_at_50": 51.0,
            }
        )
    )
    (base / "details.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"uid": "U1", "first_hit_rank": 10}),
                json.dumps({"uid": "U2", "first_hit_rank": None}),
            ]
        )
        + "\n"
    )
    (head / "details.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"uid": "U1", "first_hit_rank": 5}),
                json.dumps({"uid": "U2", "first_hit_rank": None}),
            ]
        )
        + "\n"
    )

    out = tmp_path / "compare.json"
    subprocess.run(
        [
            sys.executable,
            "compare_eval_runs.py",
            str(tmp_path / "base"),
            str(tmp_path / "head"),
            "--out",
            str(out),
        ],
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    result = json.loads(out.read_text())
    assert result["retrieval"]["summary_delta"]["recall_at_10"] == 5.0
    assert result["retrieval"]["counts"]["improved"] == 1
    assert result["retrieval"]["counts"]["regressed"] == 0
