# Best Dev Results

Best single-model multitask run:

- Run name: `adv-mt-v1`
- Dev SST accuracy: `0.513170`
- Dev paraphrase accuracy: `0.827327`
- Dev STS correlation: `0.881286`
- Dev average: `0.740594`

Best final per-task combination on dev:

- SST: `predictions/adv-sst-dev-output.csv`
- Paraphrase: `predictions/para-dev-output.csv`
- STS: `predictions/adv-sts-dev-output.csv`
- Dev SST accuracy: `0.513170`
- Dev paraphrase accuracy: `0.855005`
- Dev STS correlation: `0.881286`
- Dev average: `0.749820`

Final copied prediction files:

- `predictions/final-best-sst-dev.csv`
- `predictions/final-best-sst-test.csv`
- `predictions/final-best-para-dev.csv`
- `predictions/final-best-para-test.csv`
- `predictions/final-best-sts-dev.csv`
- `predictions/final-best-sts-test.csv`

Code added for the final search:

- `advanced_multitask_classifier.py`
- `advanced_sst_classifier.py`
- `score_predictions.py`
