# What is read, and how a value becomes a fact

Fact = `{id, name, canon, value, num, scope, kind, file, line, context}`. `canon` is the canonical parameter (learning_rate, batch_size, epochs, seed, weight_decay, dropout, optimizer, scheduler, warmup, ... about 25) when the name is a known alias; `scope` is train / val / test / data / model / optim / env / general, from the key path, the call, the file name (`eval.py`) or the clause of a sentence; `kind` is where the value came from. Nothing is guessed: a value that is not written in a file is not a fact.

| Source | Reader | kind |
|---|---|---|
| `.yaml .yml .json .toml .ini .cfg .env` | own readers (no PyYAML / tomllib needed): block and flow YAML, comments, block scalars, TOML tables / arrays of tables / inline tables / multi-line strings; `1e-3` is a number; every scalar leaf with its line | `config` |
| `.py`, notebooks (`.ipynb` code cells + outputs) | `ast`: `add_argument(..., default=)` and `store_true` flags, keyword arguments of ML-looking calls (`AdamW(lr=)`, `DataLoader(batch_size=)`, `train_test_split(test_size=)`), `manual_seed(7)`, upper-case constants, dict / `dict()` configs, dataclass / `*Config` fields | `argparse`, `call`, `assign` |
| `.sh`, Makefile, README, notes | command lines (`python train.py --lr 0.001`, `torchrun`, `sbatch` ...) with flags | `cli`, `doc` |
| logs (`.log`, `.out`, JSON lines, notebook outputs) | `Namespace(...)` dumps, `key: value` parameter lines, `epoch i/N`, learning-rate series (first / last), metric lines (`val acc: 0.88`, `Test F1 0.812`), environment banners (PyTorch / CUDA / Python versions, GPU model and count, CPU), dataset sizes per split | `log`, `result`, `env` |
| `.csv .tsv` | numeric cells as results (up to 400), row count | `result`, `data` |
| `requirements*.txt`, `environment.yml`, `pyproject.toml`, `Dockerfile` | package versions, base image | `env` |
| prose (README, notes, `.md .docx .pdf .txt`, long text fields of `case.json`, task text) | sentences that name a parameter and a value ("learning rate of 3e-4", "50 epochs", "seed 42"), sizes ("12,000 images"), splits ("8:1:1", "70%/15%/15%"), metrics ("F1 为 0.812"), versions, GPU models and counts, dataset names; Chinese and English | `doc` |
| `settings` / `config` / `params` / `参数` dictionaries inside `case.json` | like a config file | `config` |

## Conflicts
Per parameter and stage (training-type parameters without a stage count as training values). Runtime evidence (log, command line) settles a value when its sources agree: configs and code defaults that differ are shown as *overridden*, not as conflicts. A conflict is reported when run logs / command lines disagree with each other, when configuration files disagree, when a config disagrees with a value hard-coded in code or notes, or when only code / notes exist and they disagree. argparse defaults never conflict with a config (the config usually overrides them) but are listed in the table as `overridden default`. Scheduler / optimizer names that contain each other (`cosine` / `CosineAnnealingLR`) are equal.

## Draft matching
Numbers in the draft are found after removing citations, `Table 2` / `Figure 3` / `Step 1` / `第3节` style references, URLs, code spans, list numbers, version strings (checked separately) and `[[CONFIRM: ...]]` markers. A number is **sourced** when it equals a fact (also `1e-3` = `0.001` = `1 x 10^-3`, `87.1%` = `0.871`, `8k` = 8000, `8:1:1` = a split fact in any form) or is derived from dataset sizes (totals, shares). When the words before or after a number name a parameter ("learning rate 0.01", "50 epochs", "4 GPUs"), it must equal a fact of that parameter, otherwise `value_mismatch` lists the values the sources give. Common values (1, 0.1, 0.5, 100 ...) equal some fact by coincidence, so without a parameter name they are accepted only if the sentence shares a word with the fact's name or context.
