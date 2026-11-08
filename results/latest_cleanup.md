## Cleanup benchmark

| backend | n | zero-edit (strict) | zero-edit (lenient) | WER | char-edit | p50 ms | p95 ms | TTFT p50 ms | fallback |
|---|---|---|---|---|---|---|---|---|---|
| raw | 27 | 0% | 19% | 28% | 38% | 0 | 0 | - | 0% |
| rules | 27 | 44% | 59% | 15% | 19% | 0 | 0 | - | 0% |

Zero-edit rate by category (lenient):

| category | raw | rules |
|---|---|---|
| correction | 0% | 43% |
| filler | 0% | 100% |
| mixed | 0% | 67% |
| names | 0% | 0% |
| punctuation | 100% | 100% |
| question | 100% | 100% |
| repetition | 0% | 25% |
