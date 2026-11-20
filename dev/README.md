# dev/ - tests and evaluations of methods-ledger (not part of the Skill runtime)

| file | what |
|---|---|
| `test_all.py` | 36 regression tests (stdlib `unittest`); `python3 dev/test_all.py [-v]` |
| `eval_check.py` | number-to-source matching on statements generated from the invented project's facts: faithful / perturbed / invented |
| `eval_extract.py` | argparse defaults of 6 public PyTorch example scripts (BSD-3-Clause, `fixtures/public`), YAML reader vs PyYAML on 27 public files (`fixtures/public_yaml`), TOML reader vs tomli on 10 public pyproject files (`fixtures/public_toml`); needs `pip install --target dev/_deps pyyaml tomli` |
| `make_examples.py` | regenerates `submission/examples/*/out` and `run.txt` from real runs |
| `make_docx.py` | builds a small .docx for tests |
| `fixtures/project` | the invented experiment project; `draft_good.md`, `draft_bad.md` are two drafts of its methods section |

The public fixtures are copies of files from open-source repositories (pytorch/examples, detectron2, Kubernetes docs, GitHub Actions workflows, yolov5, pyproject files of flask, requests, numpy ...) kept only as test data; their licences stay with their authors. Everything else is invented.
