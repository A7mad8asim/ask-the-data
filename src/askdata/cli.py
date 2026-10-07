"""Command line:  askdata generate | ask "<question>" | eval [options]"""

from __future__ import annotations

import argparse
import sys

from .config import Settings


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> None:
    _utf8_console()
    parser = argparse.ArgumentParser(prog="askdata", description="Ask-the-Data: bilingual text-to-SQL for clinic analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="build the synthetic clinic database")
    gen.add_argument("--patients", type=int, default=60_000)
    gen.add_argument("--seed", type=int, default=42)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--provider", choices=["ollama", "anthropic", "oracle"])
    ask.add_argument("--model")

    sub.add_parser("eval", help="run the evaluation (see eval/run_eval.py --help)", add_help=False)

    args, rest = parser.parse_known_args(argv)
    if args.command == "generate":
        from .generator import generate

        settings = Settings.from_env()
        print(f"Generating {args.patients:,} patients into {settings.data_dir} ...")
        generate(args.patients, args.seed, settings.data_dir)
    elif args.command == "ask":
        _ask(args)
    elif args.command == "eval":
        from .evaluation import main as eval_main

        eval_main(rest)


def _ask(args) -> None:
    from .answers import markdown_table
    from .pipeline import Assistant

    settings = Settings.from_env()
    if args.provider:
        settings = settings.with_overrides(llm_provider=args.provider)
    if args.model:
        settings = settings.with_overrides(llm_model=args.model)
    ans = Assistant(settings).ask(args.question)
    if ans.status != "ok":
        print(f"[{ans.status}] {ans.message}")
        return
    print(ans.text)
    print()
    print(markdown_table(ans.display_table))
    print()
    print(ans.sql)
    print()
    notes = [f"{ans.latency_s:.1f} s", f"{len(ans.attempts)} attempt(s)", settings.llm_provider + ":" + settings.llm_model]
    if ans.suppressed_rows:
        notes.append(f"{ans.suppressed_rows} row(s) masked as <10")
    if ans.grounded is False:
        notes.append("model wording replaced by a plain summary (numbers not in the table)")
    print(" · ".join(notes))


if __name__ == "__main__":
    main()
