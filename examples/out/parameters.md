| parameter | value | scope | status | source |
|---|---|---|---|---|
| data.name | LS-Tiles | general | single-source | config.yaml:4 |
| data.version | v2.1 | general | single-source | config.yaml:5 |
| image_size | 256 | general | single-source | config.yaml:8 |
| model | resnet18 | general | single-source | config.yaml:20 |
| num_workers | 4 | general | single-source | train.py:20 |
| precision | True | general | settled by run log / command line (logs/train.log:1) | logs/train.log:1 |
| seed | 42 | general | settled by run log / command line (logs/train.log:1) | README.md:2; config.yaml:2 |
| batch_size | 64 | test | consistent | config.yaml:18; run.sh:4 |
| batch_size | 32 (config.yaml:11) / 16 (train.py:20) | train | CONFLICT | config.yaml:11; train.py:20 |
| dropout | 0.2 (config.yaml:21) / 0.5 (train.py:22) | train | CONFLICT | config.yaml:21; train.py:22 |
| epochs | 50 | train | settled by run log / command line (logs/train.log) | config.yaml:10; logs/train.log |
| learning_rate | 0.001 | train | settled by run log / command line (logs/train.log:1) | config.yaml:13; logs/train.log:1 |
| optimizer | AdamW | train | consistent | config.yaml:12; train.py:19 |
| scheduler | cosine | train | consistent | config.yaml:15; train.py:21 |
| warmup | 500 | train | single-source | config.yaml:16 |
| weight_decay | 0.01 (config.yaml:14) / 0.05 (train.py:19) | train | CONFLICT | config.yaml:14; train.py:19 |
