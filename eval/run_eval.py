"""Run the evaluation. See src/askdata/evaluation.py for what is measured and how.

    python eval/run_eval.py                       # full pipeline with the provider in .env
    python eval/run_eval.py --ablation            # schema -> + glossary -> + few-shot -> + repair
    python eval/run_eval.py --provider anthropic  # Claude API comparison
    python eval/run_eval.py --provider oracle     # sanity check of the harness (expects 100%)
"""

from askdata.evaluation import main

if __name__ == "__main__":
    main()
