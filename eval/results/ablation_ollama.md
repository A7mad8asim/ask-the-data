| Version | Model | Execution accuracy (EN) | Execution accuracy (AR) | Overall |
| --- | --- | --- | --- | --- |
| Schema only | ollama:qwen3:8b | 54.0% | 46.0% | 50.0% |
| + bilingual glossary | ollama:qwen3:8b | 62.0% | 40.0% | 51.0% |
| + retrieved few-shot examples | ollama:qwen3:8b | 82.0% | 64.0% | 73.0% |
| + repair loop (full pipeline) | ollama:qwen3:8b | 84.0% | 74.0% | 79.0% |
