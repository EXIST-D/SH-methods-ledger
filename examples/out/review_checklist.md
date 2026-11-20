# Review checklist

Machine-assisted draft: nothing here is approved by the author.

- [ ] open_conflict : sources disagree on batch_size [train]: 32 @ config.yaml:11  vs  16 @ train.py:20
- [ ] open_conflict : sources disagree on weight_decay [train]: 0.01 @ config.yaml:14  vs  0.05 @ train.py:19
- [ ] open_conflict : sources disagree on dropout [train]: 0.2 @ config.yaml:21  vs  0.5 @ train.py:22

## Always confirm by hand
- every value marked CONFLICT or `[[CONFIRM: ...]]`
- that the comparison conditions stated for the baselines were really the same
- values the files cannot show (set at run time, by environment variable, by an unshown code path)
