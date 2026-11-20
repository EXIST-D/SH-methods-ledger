# Evidence brief (generated; every value below is written in a file, with its source)

## Task
- language: en | length hint: none
- no task text found

## Files read (7)
- README.md [doc]: 6 facts
- config.yaml [config]: 19 facts
- requirements.txt [env]: 3 facts
- results.csv [table]: 5 facts
- run.sh [script]: 4 facts
- train.py [code]: 13 facts
- logs/train.log [log]: 25 facts

## Parameters (resolved value, scope, status, sources)
- data.name = LS-Tiles [general] single-source | config.yaml:4
- data.version = v2.1 [general] single-source | config.yaml:5
- image_size = 256 [general] single-source | config.yaml:8
- model = resnet18 [general] single-source | config.yaml:20
- num_workers = 4 [general] single-source | train.py:20
- precision = True [general] settled by run log / command line (logs/train.log:1) | logs/train.log:1
- seed = 42 [general] settled by run log / command line (logs/train.log:1) | README.md:2; config.yaml:2; logs/train.log:1
- batch_size = 64 [test] consistent | config.yaml:18; run.sh:4
- batch_size = 32 (config.yaml:11) | 16 (train.py:20) [train] CONFLICT | config.yaml:11; train.py:20
- dropout = 0.2 (config.yaml:21) | 0.5 (train.py:22) [train] CONFLICT | config.yaml:21; train.py:22
- epochs = 50 [train] settled by run log / command line (logs/train.log) | config.yaml:10; logs/train.log; logs/train.log:1
- learning_rate = 0.001 [train] settled by run log / command line (logs/train.log:1) | config.yaml:13; logs/train.log:1; run.sh:3
- optimizer = AdamW [train] consistent | config.yaml:12; train.py:19
- scheduler = cosine [train] consistent | config.yaml:15; train.py:21
- warmup = 500 [train] single-source | config.yaml:16
- weight_decay = 0.01 (config.yaml:14) | 0.05 (train.py:19) [train] CONFLICT | config.yaml:14; train.py:19

## Conflicts between sources (3 open; decide with `=== resolve`)
- batch_size [train]: 32 @ config.yaml:11  vs  16 @ train.py:20 | a configuration file and a value fixed in code / notes disagree
- weight_decay [train]: 0.01 @ config.yaml:14  vs  0.05 @ train.py:19 | a configuration file and a value fixed in code / notes disagree
- dropout [train]: 0.2 @ config.yaml:21  vs  0.5 @ train.py:22 | a configuration file and a value fixed in code / notes disagree

## Datasets and environment
- dataset.size.tile = 10000 | README.md:2
- dataset.size.region = 14 | README.md:2
- dataset.split_ratio = 8:1:1 | README.md:2
- data.name = LS-Tiles | config.yaml:4
- data.version = v2.1 | config.yaml:5
- data.root = /data/ls_tiles | config.yaml:6
- data.split = [0.8, 0.1, 0.1] | config.yaml:7
- data.split[0] = 0.8 | config.yaml:7
- data.split[1] = 0.1 | config.yaml:7
- data.split[2] = 0.1 | config.yaml:7
- dataset.size.train = 8000 | logs/train.log:4
- dataset.size.val = 1000 | logs/train.log:4
- dataset.size.test = 1000 | logs/train.log:4
- table_rows.results.csv = 2 | results.csv
- hardware.gpu = NVIDIA A100 | README.md:3
- hardware.num_gpus = 4 | README.md:3
- env.torch = 2.1.0 | requirements.txt:1
- env.numpy = 1.26.4 | requirements.txt:2
- env.scikit-learn = >=1.3 | requirements.txt:3
- env.pytorch = 2.1.0 | logs/train.log:2
- env.cuda = 12.1 | logs/train.log:2
- env.python = 3.10.12 | logs/train.log:2
- hardware.gpu = NVIDIA A100-SXM4-40GB | logs/train.log:3

## Results found (last value per metric)
- F1 = 0.771 | results.csv:3 (2 values)
- params_M = 11.2 | results.csv:3 (2 values)
- val_acc = 0.88 | logs/train.log:7 (3 values)
- train_loss = 0.201 | logs/train.log:7 (3 values)
- test_accuracy = 0.871 | logs/train.log:8
- test_f1 = 0.812 | logs/train.log:9

## Commands found (2)
- run.sh:3 `python train.py --config config.yaml --epochs 50 --lr 0.001 --seed 42`
- run.sh:4 `python eval.py --ckpt best.pt --batch-size 64`

## Not in the materials (4): ask the author, never fill in
- setup.repeats: How many independent runs / folds were done, or was it a single run? Results reported as mean ± std need the count.
- baseline.statistics: Were results averaged over runs? Please give mean ± std, the number of runs and any significance test.
- repro.code_version: Which code version (git commit hash or release tag) produced the results?
- repro.data_access: Where can the data be obtained (URL, DOI, access conditions)?
