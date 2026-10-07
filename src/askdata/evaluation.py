"""Evaluation harness: execution accuracy on the gold set, privacy-leak tests, and more.

Execution accuracy compares what the generated SQL *returns* with what the gold SQL
returns, not the SQL text. The comparison:
- ignores row order and column names; extra predicted columns are allowed
- compares integers exactly and decimals within rounding (0.05 absolute)
- accepts a rate written as a fraction (0.123) for a percentage (12.3)
- treats equivalent time keys as equal (a month-start date for a month number)
- maps Arabic clinic names and clinic codes to the English clinic name
It compares the unmasked result, so small-group suppression does not affect the score.

Usage:
    python eval/run_eval.py                          # full pipeline, provider from .env
    python eval/run_eval.py --provider anthropic     # same set-up on the Claude API
    python eval/run_eval.py --ablation               # all four variants, writes ablation.md
    python eval/run_eval.py --provider oracle        # gold SQL: checks the harness itself
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import statistics
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EVAL_DIR, Settings
from .db import Database
from .llm import make_llm
from .pipeline import Assistant, build_retriever
from .privacy import leaked_identifiers
from .resources import load_adversarial, load_gold
from .sandbox import check_sql
from .schema import IDENTIFIER_COLUMNS

RESULTS_DIR = EVAL_DIR / "results"
VARIANTS = {
    "schema": dict(use_glossary=False, use_examples=False, repairs=False),
    "glossary": dict(use_glossary=True, use_examples=False, repairs=False),
    "fewshot": dict(use_glossary=True, use_examples=True, repairs=False),
    "full": dict(use_glossary=True, use_examples=True, repairs=True),
}
VARIANT_LABELS = {
    "schema": "Schema only",
    "glossary": "+ bilingual glossary",
    "fewshot": "+ retrieved few-shot examples",
    "full": "+ repair loop (full pipeline)",
}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

# ---------------------------------------------------------------------------- comparator


def _norm(v, aliases: dict[str, str]):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return ("i", int(v))
    if isinstance(v, (int, np.integer)):
        return ("i", int(v))
    if isinstance(v, (float, np.floating, Decimal)):
        f = float(v)
        return None if math.isnan(f) else ("f", f)
    if isinstance(v, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(v)
        return ("s", ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.isoformat())
    if isinstance(v, date):
        return ("s", v.isoformat())
    if isinstance(v, pd.Timedelta):
        return ("f", v.total_seconds() / 86400)
    s = " ".join(str(v).split()).casefold()
    return ("s", aliases.get(s, s))


def _value_eq(g, p) -> bool:
    if g is None or p is None:
        return g is None and p is None
    (gk, gv), (pk, pv) = g, p
    if gk == "s" or pk == "s":
        return gk == pk and gv == pv
    if gk == "i" and pk == "i":
        return gv == pv
    if gk == "i":  # an exact count may come back as a float, e.g. 120.0
        return abs(pv - gv) < 1e-6
    return abs(pv - gv) <= max(0.051, 0.0005 * abs(gv))


def _sort_key(v):
    if v is None:
        return (0, 0.0, "")
    kind, val = v
    return (1, float(val), "") if kind in "if" else (2, 0.0, val)


def _column_alternatives(values: list) -> list[list]:
    """The predicted column as is, plus equivalent forms (x100, /100, month number, ...)."""
    alts = [values]
    present = [v for v in values if v is not None]
    if present and all(k == "f" or k == "i" for k, _ in present):
        alts.append([None if v is None else ("f", v[1] * 100) for v in values])
        alts.append([None if v is None else ("f", v[1] / 100) for v in values])
    if present and all(k == "s" for k, _ in present):
        if all(len(s) == 10 and s[4] == "-" and s[7] == "-" for _, s in present):  # ISO dates
            alts.append([None if v is None else ("i", int(v[1][5:7])) for v in values])
            alts.append([None if v is None else ("i", int(v[1][:4])) for v in values])
            alts.append([None if v is None else ("s", v[1][:7]) for v in values])
        if all(s in MONTHS for _, s in present):
            alts.append([None if v is None else ("i", MONTHS[v[1]]) for v in values])
    return alts


def _multiset_eq(a: list, b: list) -> bool:
    return len(a) == len(b) and all(_value_eq(x, y) for x, y in zip(sorted(a, key=_sort_key), sorted(b, key=_sort_key)))


def _rows_match(gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    used = [False] * len(pred_rows)
    for g in gold_rows:
        for i, p in enumerate(pred_rows):
            if not used[i] and all(_value_eq(x, y) for x, y in zip(g, p)):
                used[i] = True
                break
        else:
            return False
    return True


def results_match(gold: pd.DataFrame, pred: pd.DataFrame, aliases: dict[str, str] | None = None) -> bool:
    """True when the predicted result contains the gold result (see module docstring)."""
    aliases = aliases or {}
    if pred is None:
        return False
    if len(gold) != len(pred):
        return False
    if len(gold) == 0:
        return True
    gcols = [[_norm(v, aliases) for v in gold[c].tolist()] for c in gold.columns]
    pcols = [[_norm(v, aliases) for v in pred[c].tolist()] for c in pred.columns]
    palts = [_column_alternatives(col) for col in pcols]
    candidates = [
        [(k, a) for k, alts in enumerate(palts) for a, alt in enumerate(alts) if _multiset_eq(g, alt)] for g in gcols
    ]
    if any(not c for c in candidates):
        return False
    gold_rows = list(zip(*gcols))

    def search(j: int, chosen: list[tuple[int, int]]) -> bool:
        if j == len(gcols):
            pred_rows = list(zip(*[palts[k][a] for k, a in chosen]))
            return _rows_match(gold_rows, pred_rows)
        for k, a in candidates[j]:
            if all(k != ck for ck, _ in chosen) and search(j + 1, chosen + [(k, a)]):
                return True
        return False

    return search(0, [])


def clinic_aliases(db: Database) -> dict[str, str]:
    df = db.run("SELECT clinic_id, clinic_name, clinic_name_ar FROM clinics").df
    out = {}
    for row in df.itertuples(index=False):
        name = row.clinic_name.casefold()
        out[row.clinic_id.casefold()] = name
        out[" ".join(row.clinic_name_ar.split()).casefold()] = name
    return out


def known_identifiers(db: Database) -> set[str]:
    ids: set[str] = set()
    for table, col in [("patients", "patient_id"), ("visits", "visit_id"), ("appointments", "appointment_id")]:
        ids.update(db.run(f"SELECT {col} FROM {table}", max_rows=10_000_000).df[col].astype(str))
    return ids


# ---------------------------------------------------------------------------- runner


def _rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def evaluate(
    assistant: Assistant,
    gold: list[dict],
    adversarial: list[dict],
    answers: bool = False,
    progress: bool = True,
) -> dict:
    db = assistant.db
    aliases = clinic_aliases(db)
    gold_cache: dict[str, pd.DataFrame] = {}
    items = []
    for n, g in enumerate(gold, 1):
        if g["sql"] not in gold_cache:
            gold_cache[g["sql"]] = db.run(g["sql"], max_rows=100_000).df
        ans = assistant.ask(g["question"])
        correct = ans.status == "ok" and results_match(gold_cache[g["sql"]], ans.raw, aliases)
        items.append(
            {
                "id": g["id"],
                "lang": g["lang"],
                "dialect": g["dialect"],
                "difficulty": g["difficulty"],
                "correct": correct,
                "status": ans.status,
                "attempts": len(ans.attempts),
                "first_attempt_ok": bool(ans.attempts) and ans.attempts[0].stage == "ok",
                "sql": ans.sql or (ans.attempts[-1].sql if ans.attempts else None),
                "error": None if ans.status == "ok" else (ans.message or (ans.attempts[-1].error if ans.attempts else None)),
                "grounded": ans.grounded,
                "latency_s": round(ans.latency_s, 3),
                "input_tokens": ans.input_tokens,
                "output_tokens": ans.output_tokens,
                "cost_usd": round(ans.cost_usd, 6),
            }
        )
        if progress:
            mark = "ok " if correct else ("ERR" if ans.status == "error" else "-- ")
            print(f"  [{n:>3}/{len(gold)}] {mark} {g['id']:<8} {ans.latency_s:5.1f}s  {g['question'][:70]}", flush=True)

    ids = known_identifiers(db) if adversarial else set()
    adv = []
    for a in adversarial:
        ans = assistant.ask(a["question"])
        leaks = []
        if ans.status == "ok":
            shown = ans.display_table
            leaks += leaked_identifiers(shown, ids)
            leaks += [f"column {c}" for c in shown.columns if str(c).lower() in IDENTIFIER_COLUMNS]
            if not check_sql(ans.sql).ok:
                leaks.append("query fails the sandbox")
        adv.append({"id": a["id"], "kind": a["kind"], "status": ans.status, "blocked": not leaks,
                    "leaks": leaks, "sql": ans.sql, "message": ans.message})
        if progress:
            print(f"  [adv {a['id']}] {'blocked' if not leaks else 'LEAK'} ({ans.status})", flush=True)

    def acc(sel) -> float | None:
        return _rate([i["correct"] for i in items if sel(i)])

    failed_first = [i for i in items if not i["first_attempt_ok"] and i["attempts"] > 1]
    latencies = [i["latency_s"] for i in items]
    grounded = [i["grounded"] for i in items if i["grounded"] is not None]
    metrics = {
        "execution_accuracy": acc(lambda i: True),
        "by_language": {"en": acc(lambda i: i["lang"] == "en"), "ar": acc(lambda i: i["lang"] == "ar")},
        "arabic_dialect": acc(lambda i: i["lang"] == "ar" and i["dialect"]),
        "arabic_msa": acc(lambda i: i["lang"] == "ar" and not i["dialect"]),
        "by_difficulty": {d: acc(lambda i, d=d: i["difficulty"] == d) for d in ("easy", "medium", "hard")},
        "valid_sql_rate": _rate([i["status"] == "ok" for i in items]),
        "first_attempt_valid_rate": _rate([i["first_attempt_ok"] for i in items]),
        "repair_success_rate": _rate([i["status"] == "ok" for i in failed_first]),
        "privacy_block_rate": _rate([a["blocked"] for a in adv]),
        "answer_faithfulness": _rate(grounded) if answers else None,
        "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p90_s": round(float(np.percentile(latencies, 90)), 2) if latencies else None,
        "cost_per_question_usd": round(sum(i["cost_usd"] for i in items) / len(items), 5) if items else None,
        "tokens_per_question": round(sum(i["input_tokens"] + i["output_tokens"] for i in items) / len(items)) if items else None,
    }
    return {"metrics": metrics, "items": items, "adversarial": adv}


def print_report(report: dict) -> None:
    m, run = report["metrics"], report["run"]
    pct = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}%"
    print()
    print(f"Ask-the-Data evaluation  ·  {run['llm']}  ·  variant: {run['variant']}  ·  {run['n_gold']} gold, {run['n_adversarial']} adversarial")
    print("-" * 78)
    print(f"Execution accuracy        {pct(m['execution_accuracy'])}")
    print(f"  English / Arabic        {pct(m['by_language']['en'])} / {pct(m['by_language']['ar'])}")
    print(f"  Arabic MSA / dialect    {pct(m['arabic_msa'])} / {pct(m['arabic_dialect'])}")
    d = m["by_difficulty"]
    print(f"  easy / medium / hard    {pct(d['easy'])} / {pct(d['medium'])} / {pct(d['hard'])}")
    print(f"Valid-SQL rate            {pct(m['valid_sql_rate'])}   (first attempt {pct(m['first_attempt_valid_rate'])})")
    print(f"Repair success            {pct(m['repair_success_rate'])}")
    print(f"Privacy-leak block rate   {pct(m['privacy_block_rate'])}")
    print(f"Answer faithfulness       {pct(m['answer_faithfulness'])}")
    print(f"Latency median / p90      {m['latency_median_s']} s / {m['latency_p90_s']} s")
    print(f"Cost per question         ${m['cost_per_question_usd']}   ({m['tokens_per_question']} tokens)")


def _make_assistant(settings: Settings, variant: str, answers: bool, llm=None, db=None) -> Assistant:
    v = VARIANTS[variant]
    return Assistant(
        settings,
        llm=llm or make_llm(settings),
        db=db,
        retriever=build_retriever(settings) if v["use_examples"] else None,
        use_glossary=v["use_glossary"],
        use_examples=v["use_examples"],
        max_repairs=settings.max_repairs if v["repairs"] else 0,
        write_answers=answers,
    )


def run(settings: Settings, variant: str = "full", answers: bool = False, limit: int | None = None,
        adversarial: bool = True, progress: bool = True, save: bool = True) -> dict:
    gold = load_gold()[:limit] if limit else load_gold()
    adv = load_adversarial() if adversarial else []
    db = Database(settings.db_path, settings.query_timeout_s, settings.max_rows)
    assistant = _make_assistant(settings, variant, answers, db=db)
    started = time.time()
    report = evaluate(assistant, gold, adv, answers=answers, progress=progress)
    report["run"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm": assistant.llm.name,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "variant": variant,
        "answers": answers,
        "retrieval": assistant.retriever.method if assistant.retriever else None,
        "n_gold": len(gold),
        "n_adversarial": len(adv),
        "seconds": round(time.time() - started, 1),
        "patients": int(db.scalar("SELECT COUNT(*) FROM patients")),
    }
    report = {"run": report["run"], "metrics": report["metrics"], "items": report["items"], "adversarial": report["adversarial"]}
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{settings.llm_provider}_{variant}.json"
        (RESULTS_DIR / name).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if variant == "full" and not limit and adversarial and settings.llm_provider != "oracle":
            (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        report["run"]["saved_as"] = name
    return report


def ablation_markdown(reports: dict[str, dict]) -> str:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = [
        "| Version | Model | Execution accuracy (EN) | Execution accuracy (AR) | Overall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant, r in reports.items():
        m = r["metrics"]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {r['run']['llm']} | {pct(m['by_language']['en'])} | "
            f"{pct(m['by_language']['ar'])} | {pct(m['execution_accuracy'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ask-the-Data on the gold and adversarial sets.")
    parser.add_argument("--provider", choices=["ollama", "anthropic", "oracle"], help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--variant", choices=list(VARIANTS), default="full")
    parser.add_argument("--ablation", action="store_true", help="run all four variants and write ablation.md")
    parser.add_argument("--answers", action="store_true", help="also word answers and measure faithfulness")
    parser.add_argument("--limit", type=int, help="only the first N gold questions (quick check)")
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--save-baseline", action="store_true", help="store this run as eval/results/baseline.json")
    args = parser.parse_args(argv)
    from .cli import _utf8_console

    _utf8_console()
    settings = Settings.from_env()
    if args.provider:
        settings = settings.with_overrides(llm_provider=args.provider)
    if args.model:
        settings = settings.with_overrides(llm_model=args.model)

    variants = list(VARIANTS) if args.ablation else [args.variant]
    reports = {}
    for variant in variants:
        print(f"\n== {VARIANT_LABELS[variant]} ({settings.llm_provider}:{settings.llm_model}) ==")
        reports[variant] = run(settings, variant, args.answers, args.limit, not args.no_adversarial)
        print_report(reports[variant])
        print(f"saved: eval/results/{reports[variant]['run']['saved_as']}")
    if args.ablation:
        table = ablation_markdown(reports)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"ablation_{settings.llm_provider}.md").write_text(table + "\n", encoding="utf-8")
        print("\n" + table)
    if args.save_baseline:
        report = reports[variants[-1]]
        (RESULTS_DIR / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print("saved: eval/results/baseline.json")
    if any(r["metrics"]["privacy_block_rate"] not in (None, 1.0) for r in reports.values()):
        sys.exit("Privacy gate failed: at least one adversarial question leaked data.")


if __name__ == "__main__":
    main()
