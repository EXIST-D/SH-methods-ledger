# Checks (`check` / `build`)

Severity: **error** = fix before handing over (`build` still writes everything, `status` = `partial`); **warn** = fix or justify; **info** = shown with `--all`.

## Numbers and names (every sentence)
| Code | Sev | Test |
|---|---|---|
| `unsourced_number` | error | a number (or split ratio, or a parameter value with its name) equals no fact and is not derivable from dataset sizes; small whole counts (≤ 12 without a decimal point), years and a number coinciding with a fact that shares no word with the sentence are the only unchecked ones |
| `value_mismatch` | error | the words around a number name a parameter and the sources give another value; also optimizer name, package version (PyTorch / CUDA / Python ...), GPU model, dataset version |
| `condition_mixup` | error | the value is the training one but the sentence describes evaluation (or the reverse), and the sources give another value for the stage described |
| `unsourced_name` / `unsourced_version` | error | optimizer / GPU / package version named without any source (a dataset version without source is a warning) |
| `conflicted_value_used` | warn | the value is used although its sources disagree and the sentence carries no `[[CONFIRM]]` |
| `placeholder_left` | error | `XX`, `TODO`, `TBD`, `???`, `待补` left in the text |
| `unverified_comparability` | warn | "same budget", "fair comparison", "同等预算" with no source on the baselines' budget |
| `compressed_steps` | info | one long sentence with three or more sequence words (then / next / finally / 然后 ...) |

## Elements (once per draft)
| Code | Sev | Test |
|---|---|---|
| `element_omitted` | warn | the materials hold the element (seed, optimizer, hardware, versions, dataset size / split / preprocessing, metrics, commands ...) but the text never gives it |
| `element_without_source` | warn | the text mentions an element no file documents |
| `gap_not_marked` | warn | an element is in neither the materials nor the text as `[[CONFIRM: ...]]` |
| `needs_author_confirmation` | info | every `[[CONFIRM: ...]]` in the text (also written to `questions.md`) |
| `open_conflict` | warn | a conflict between sources not settled by `=== resolve` |
| `no_limitations`, `generic_limitation` | warn | no limitation stated; a limitation without a failure scenario ("may not generalize", "further research is needed" without a condition, number or setting) |
| `length` | warn | outside 70-140% of the task's length hint |

Elements checked (18): seed, repeats, epochs / steps, batch size, optimizer, learning rate and schedule, hardware, software versions, dataset name, version, size, split, preprocessing, metrics, statistics, code version, commands, data access.

## Measured behaviour (development data; the fixture was used while the rules were written, so these are optimistic)
- `dev/eval_check.py` (invented project, 22 parameter statements in English and Chinese, generated from its facts): faithful statements accepted without an error 22 / 22; the same statements with the value changed flagged 21 / 22 (the miss: halving 64 gives 32, which really is the training value); statements about 6 parameters with no source anywhere flagged 6 / 6.
- `dev/eval_extract.py`: argparse defaults of 6 public PyTorch example scripts recovered with the same value 65 / 65; YAML reader vs PyYAML on 27 public files (detectron2, Kubernetes, GitHub Actions, docker-compose, yolov5 ...) identical scalar leaves 27 / 27; TOML reader vs tomli on 10 public `pyproject.toml` files 8 / 10 (differences: escape handling in literal / line-continued strings).
- `dev/test_all.py`: 36 tests (readers, extractors, conflicts, draft checks incl. Chinese, ledger rejections, build and contract).
Limits: values are read from text; a value computed at run time, set by an environment variable or by code that the files do not show is invisible. The element check is lexical (a phrase in the text), not semantic.
