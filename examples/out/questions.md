# Questions for the author (the materials do not answer these)

1. [conflict] Which value is right for batch_size [train]? 32 in config.yaml:11  vs  16 in train.py:20. (a configuration file and a value fixed in code / notes disagree)
2. [conflict] Which value is right for weight_decay [train]? 0.01 in config.yaml:14  vs  0.05 in train.py:19. (a configuration file and a value fixed in code / notes disagree)
3. [conflict] Which value is right for dropout [train]? 0.2 in config.yaml:21  vs  0.5 in train.py:22. (a configuration file and a value fixed in code / notes disagree)
4. [missing] How many independent runs / folds were done, or was it a single run? Results reported as mean ± std need the count.
5. [missing] Were results averaged over runs? Please give mean ± std, the number of runs and any significance test.
6. [missing] Which code version (git commit hash or release tag) produced the results?
7. [missing] Where can the data be obtained (URL, DOI, access conditions)?
8. [marked_in_text] weight decay - 0.01 in config.yaml, 0.05 hard-coded in the AdamW call of train.py
9. [marked_in_text] training batch size - config.yaml says 32, the DataLoader call in train.py says 16
10. [marked_in_text] dropout - 0.2 in config.yaml, 0.5 in train.py
11. [marked_in_text] how the split was drawn
12. [marked_in_text] baseline budget and training settings
13. [marked_in_text] number of repeated runs and statistics (mean and std)
14. [marked_in_text] code version (commit)
15. [marked_in_text] how to obtain LS-Tiles
