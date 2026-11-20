# Writing guide: section recipes, question templates, failure patterns

## Experimental setup (one paragraph per stage)
Training: data and split used, epochs / steps, optimizer with its hyper-parameters, learning rate and schedule, batch size, regularisation, precision, seeds, number of runs, hardware, software versions. Validation: what it selects (checkpoint, early stopping). Test / evaluation: batch size, preprocessing, metrics and how they are computed, checkpoint evaluated. Write the values as in the parameter table (`parameters.md`), the table and the prose must agree.

## Dataset
Name and version, size and per-split sizes, preprocessing and augmentation (with parameters), split proportions and how they were drawn (random with seed / by region / by time / stratified), what each subset is for, how to obtain it.

## Method steps
For each step: input, operation (with its parameters), output, decision rule. "We preprocess, train, evaluate and select the best model" is four steps in one sentence; write four.

## Baselines
Name, version or source, how it was run (same data, same budget, same tuning?), metrics, number of runs, statistics. A comparison is a statement about conditions: state them or say they are unknown.

## Reproducibility
Commands (from `commands.md`), code version (commit hash or tag), environment (requirements / container), data access, expected runtime and hardware, random seeds.

## Limitations
Each limitation names a scenario: the input, size or range for which the method or the comparison was not tested or fails, and what is at risk. Never "may not generalize" alone.

## Question templates (what `questions.md` asks the author)
Seed(s) for training and for the split; number of runs / folds and mean ± std; epochs or steps and which checkpoint is reported; training and evaluation batch size; optimizer hyper-parameters; learning rate schedule; GPU / CPU model and count; framework / CUDA / Python versions; dataset version / DOI / date; split proportions and how drawn; preprocessing and augmentation; metrics and values for the method and each baseline; baseline budget parity; code commit; commands; data access. A conflict becomes: "Which value is right for X? a in file1 vs b in file2 (runtime evidence shows c)".

## Failure patterns of unassisted drafts and the check that catches each
| Pattern | Check |
|---|---|
| a plausible value written because the paragraph needs one | `unsourced_number`, `unsourced_name` |
| the value of the config reported although the run used another | `value_mismatch` (runtime evidence settles), `open_conflict` |
| training value reported for evaluation | `condition_mixup` |
| wrong framework / GPU / dataset version | `value_mismatch`, `unsourced_version` |
| seed, repeats, hardware, versions silently left out | `element_omitted`, `gap_not_marked` |
| "fair comparison" without a source | `unverified_comparability` |
| "may not generalize" | `generic_limitation` |
| four steps in one sentence | `compressed_steps` |
| prose and parameter table disagree | `source_map.csv`, `parameters.md` built from the same facts |
