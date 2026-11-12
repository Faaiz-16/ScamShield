# Verdict evaluation

Test set: `/Users/faaizali/PROJECTS/ScamShield/data/processed/test.csv` (25 messages). Thresholds: suspicious 0.35, likely scam 0.65. A scam counts as caught when the verdict is Likely Scam; flagged means Suspicious or Likely Scam.

| Classifier | Likely Scam precision | Likely Scam recall | Likely Scam F1 | Non-scams called Likely Scam | Scams flagged | Non-scams flagged |
|---|---|---|---|---|---|---|
| none | 1.000 | 0.750 | 0.857 | 0.0% | 91.7% | 0.0% |
| baseline | 1.000 | 0.917 | 0.957 | 0.0% | 91.7% | 15.4% |
| muril | 1.000 | 0.917 | 0.957 | 0.0% | 91.7% | 15.4% |

## none

| True label | Safe | Suspicious | Likely Scam |
|---|---|---|---|
| legit | 10 | 0 | 0 |
| scam | 1 | 2 | 9 |
| spam_not_scam | 3 | 0 | 0 |

Likely Scam threshold sweep:

| Threshold | Precision | Recall | F1 | Non-scams called Likely Scam |
|---|---|---|---|---|
| 0.40 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.45 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.50 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.55 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.60 | 1.000 | 0.833 | 0.909 | 0.0% |
| 0.65 (current) | 1.000 | 0.750 | 0.857 | 0.0% |
| 0.70 | 1.000 | 0.750 | 0.857 | 0.0% |
| 0.75 | 1.000 | 0.667 | 0.800 | 0.0% |
| 0.80 | 1.000 | 0.667 | 0.800 | 0.0% |
| 0.85 | 1.000 | 0.667 | 0.800 | 0.0% |
| 0.90 | 1.000 | 0.417 | 0.588 | 0.0% |

Best F1 at threshold 0.55.

## baseline

| True label | Safe | Suspicious | Likely Scam |
|---|---|---|---|
| legit | 10 | 0 | 0 |
| scam | 1 | 0 | 11 |
| spam_not_scam | 1 | 2 | 0 |

Likely Scam threshold sweep:

| Threshold | Precision | Recall | F1 | Non-scams called Likely Scam |
|---|---|---|---|---|
| 0.40 | 0.846 | 0.917 | 0.880 | 15.4% |
| 0.45 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.50 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.55 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.60 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.65 (current) | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.70 | 1.000 | 0.833 | 0.909 | 0.0% |
| 0.75 | 1.000 | 0.750 | 0.857 | 0.0% |
| 0.80 | 1.000 | 0.750 | 0.857 | 0.0% |
| 0.85 | 1.000 | 0.667 | 0.800 | 0.0% |
| 0.90 | 1.000 | 0.583 | 0.737 | 0.0% |

Best F1 at threshold 0.65.

## muril

| True label | Safe | Suspicious | Likely Scam |
|---|---|---|---|
| legit | 10 | 0 | 0 |
| scam | 1 | 0 | 11 |
| spam_not_scam | 1 | 2 | 0 |

Likely Scam threshold sweep:

| Threshold | Precision | Recall | F1 | Non-scams called Likely Scam |
|---|---|---|---|---|
| 0.40 | 0.917 | 0.917 | 0.917 | 7.7% |
| 0.45 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.50 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.55 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.60 | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.65 (current) | 1.000 | 0.917 | 0.957 | 0.0% |
| 0.70 | 1.000 | 0.833 | 0.909 | 0.0% |
| 0.75 | 1.000 | 0.750 | 0.857 | 0.0% |
| 0.80 | 1.000 | 0.750 | 0.857 | 0.0% |
| 0.85 | 1.000 | 0.667 | 0.800 | 0.0% |
| 0.90 | 1.000 | 0.500 | 0.667 | 0.0% |

Best F1 at threshold 0.65.

A threshold chosen on this test set should be confirmed on fresh data before changing SCAM_THRESHOLD, otherwise the reported scores become optimistic.
