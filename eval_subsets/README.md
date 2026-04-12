Fixed UID manifests for reproducible evaluation subsets.

- `smoke_uids.json`: small current-main sanity run.
- `extraction_dev_uids.json`: fixed 50-question subset for slower live extraction evals.
- `retrieval_tune_uids.json`: seeded 100-question tuning split for hill-climbing retrieval changes.
- `retrieval_holdout_uids.json`: seeded 100-question holdout split; do not tune on this.
- `retrieval_audit_uids.json`: remaining 46 questions for spot checks and difficult-case review.

`retrieval_dev` and `full_eval` currently mean all rows from `officeqa_full.csv`,
captured in each run's `uids.json` plus the CSV hash in `manifest.json`.
