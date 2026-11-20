---
name: methods-ledger
description: Write the Methods / Experimental setup / Datasets / Baselines / Reproducibility / Limitations sections of a paper (English or Chinese) from the experiment's own files, and prove every number in them. Reads configuration files (YAML, JSON, TOML, INI, .env), Python code and notebooks (argparse defaults, call keywords, constants), shell scripts, logs, CSV result tables, requirement / Docker files, lab notes and the task text, keeps each value with its file and line, lists conflicts between sources, and checks your draft - every number and parameter name tied to a source, values that differ from the sources, training and evaluation values mixed up, invented numbers, missing elements (seed, repeats, hardware, versions, dataset version and split, baselines, commands, code version, data access), generic limitations. Builds the section (md / docx / tex), a parameter table with sources, dataset / environment tables, the commands, the questions for the author, a source map and result.json. Use for "write the methods / experiments section", "补写方法与实验设置", "整理参数表", "check my experiment description against the config / logs", "make the experiment reproducible". Python 3.9+ standard library only.
---

# methods-ledger

A methods section fails in four ways: it leaves things out, it states numbers that the config, log or code do not show (a value invented to make the paragraph read smoothly), it mixes training and evaluation conditions, and it hides what nobody knows. This skill reads the experiment's own files, keeps every value with its source, and checks your prose against them. You write; `scripts/mw.py` proves.

Not for: results discussion, figures, literature review, inventing missing details (it never fills a value that no file states).

```bash
S=/skills/weekly-skill/scripts/mw.py      # or the scripts/ folder next to this SKILL.md
```
Budget (64 requests, 900 s): `run` (1), your section (1-2 writes), `check` (1-3), `build` (1).

## Procedure

1. **Read the task** (`/input/task.md`, `/input/case.json`): language, length, which sections, audience. Task wording beats the defaults below.
2. **Read the materials in one call:**
   ```bash
   python3 $S run --case /input/case.json --out /app/output
   ```
   (or `run --src PROJECT_DIR [--file extra.md] --out /app/output`). It walks the case folder (code, configs, logs, tables, notes, requirement files; up to 5 levels), reads long text fields and settings dictionaries of `case.json`, and prints `brief.md`: files read, parameters (resolved value, stage, status, `file:line`), conflicts between sources, datasets and environment, results found, commands, and what the materials do not contain. **Read it all.** An existing draft of the section in the folder is checked automatically (it is never used as evidence). `show W --facts PATTERN` lists facts with their context, `show W --conflicts`, `show W --results`.
3. **Settle conflicts only with evidence.** Runtime evidence (a run log, the command line) already settles values that code merely defaults (`settled by run log`). What stays open (config vs a value hard-coded in code, notes vs config ...) is listed as CONFLICT: do not pick one silently. If a file or the task text lets you decide, record it:
   ```
   === resolve weight_decay train
   value: 0.01
   reason: run.sh passes config.yaml, and the AdamW call in train.py is the line the log header contradicts   # a reason is required
   === confirm repeats
   value: 5
   quote: every experiment was repeated 5 times      # VERBATIM from the task text / notes given with the case
   ```
   `python3 $S ingest /app/output/.work decisions.txt`. A `resolve` must pick an observed value and give a reason; a `confirm` needs a verbatim quote from the materials; anything else is rejected, and goes to the author as a question instead.
4. **Write the section** (Markdown, `draft.md`; `python3 $S skeleton /app/output/.work` prints a fact-bound outline with `[[CONFIRM: ...]]` where the materials are silent, if you want one to start from). Rules below.
5. **Check and repair:** `python3 $S check /app/output/.work draft.md` prints errors (fix), warnings (fix or justify), the open author confirmations, and the element table. Re-run until no error is left.
6. **Build:** `python3 $S build /app/output/.work draft.md --out /app/output [--case /input/case.json --answers answers.json]` writes `section.md/.docx/.tex`, `parameters.csv/.md`, `datasets_environment.csv`, `commands.md`, `questions.md`, `source_map.csv/json`, `review_checklist.md`, `result.json`.
7. **Fill the contract.** `result.json` takes its fields from the case's `output_contract`; derivable ones are filled, the others are `null` and listed in `_meta.unfilled_fields` and on the console as `UNFILLED`. Put your own content for them in `answers.json` (`{"field": value}`) and run `build ... --answers answers.json`. Never leave a field out, never invent its content.
8. **Before you stop:** no `OPEN ERROR` line (or each explained); every CONFLICT and `[[CONFIRM: ...]]` is reported as open in your final message; `status` is `partial` while errors or open conflicts remain — say so instead of calling the section finished.

## Rules of writing that decide quality

- **Every number comes from the brief.** Parameter values, sizes, ratios, versions, counts and results must be in the facts (or computed from dataset sizes). If a number is not there, the sentence loses it or carries `[[CONFIRM: what is needed]]`; it is never rounded into plausibility, never "typically" borrowed from the field. `check` flags `unsourced_number`, `value_mismatch` (text says lr 0.01, sources say 0.001), `unsourced_version`, `unsourced_name`.
- **Keep the conditions apart.** Say separately what training, validation and test/evaluation used (batch size, augmentation, resolution, number of runs, checkpoint chosen). `condition_mixup` fires when a training value is written for evaluation or vice versa.
- **Say what is not known, where it belongs.** The marker `[[CONFIRM: seed - not in any file]]` makes the gap visible in the text, in `questions.md` and in `result.json`. Elements the materials lack and the text does not flag are warnings (`gap_not_marked`): random seeds, repeats and statistics, hardware and software versions, dataset version, how the split was drawn, preprocessing, baseline versions and budget, metrics definition, code version (commit), commands, data access.
- **Steps as steps.** Method: an ordered sequence, each step with input, operation, output and (where there is one) the decision rule; one sentence that squeezes four steps is flagged.
- **Setup:** training / validation / test flow, hardware, software versions, seeds, repeats, optimizer and schedule with their values, in the order a reader would redo them.
- **Datasets and splits:** name, version, size, preprocessing, split proportions and how they were drawn, what each subset is used for.
- **Baselines:** what each is (version, source), whether training budget and tuning were the same (only if a source says so: `unverified_comparability` otherwise), metrics and how they are computed, statistics (mean ± std, n runs).
- **Reproducibility:** the commands (copy them from `commands.md`), code version, environment files, data access; each missing item is a `[[CONFIRM]]`, not a guess.
- **Limitations as failure scenarios:** "when the target region lies outside the 14 training regions, F1 can drop" is a limitation; "the method may not generalize" is not (`generic_limitation`). Use only scenarios the materials support or mark them as needing the author's confirmation.
- **Language and form:** the task's language; terms constant; no placeholder text (`XX`, `TODO`, `待补`): use `[[CONFIRM: ...]]`.

## Exceptions

| Situation | Do |
|---|---|
| Materials are only prose notes | numbers in notes are facts with lower trust (`doc`); conflicts between notes and the task text are reported; write only what they say |
| No config / code / logs at all | `status` stays partial; list every missing element as a question; do not describe settings from general practice |
| Conflicting values and no way to decide | keep both in the text with `[[CONFIRM: …]]` or state the value the run log shows and say where the other comes from; leave the conflict open in the report |
| Values set at run time, by environment variable or by code the files do not show | invisible to the reader tool: name them as unknown (`[[CONFIRM]]`) rather than assuming the default |
| Existing draft of the section | it is checked, never used as evidence; revise it: remove or source every flagged number |
| Several experiments / runs in one folder | settled values come from the log of the run that produced the results; if several logs disagree, the conflict is listed |
| Config file unreadable / not UTF-8 | the file is listed with its note in the brief and skipped; the others are read |
| Case without `output_contract` | `result.json` still holds status, section, sources, boundaries, files; read `task.md` for what else is wanted and add it via `--answers` |

More: `references/formats.md` (what is read and how), `references/checks.md` (every check, thresholds, measured behaviour), `references/writing-guide.md` (section recipes, failure patterns, question templates), `examples/` (invented project, flawed first draft vs repaired draft, a Chinese case). Example outputs are examples, not results of any evaluation case.
