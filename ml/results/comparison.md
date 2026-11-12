# Model comparison

Test set: `/Users/faaizali/PROJECTS/ScamShield/data/processed/test.csv` (25 messages), generated 2026-11-12T13:17:50+00:00.

## Overall

| Model | Accuracy | Macro P | Macro R | Macro F1 | Macro F1 95% CI | Scam P | Scam R | Scam F1 | Legit flagged as scam | Scam ROC-AUC | ms / message |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.600 | 0.423 | 0.444 | 0.418 | 0.269 to 0.618 | 0.556 | 0.833 | 0.667 | 50.0% | 0.737 | 0.31 |
| muril | 0.880 | 0.933 | 0.822 | 0.859 | 0.554 to 1.000 | 0.800 | 1.000 | 0.889 | 20.0% | 0.897 | 30.46 |

## By language

| Model | en macro F1 | en scam recall | hi macro F1 | hi scam recall | hinglish macro F1 | hinglish scam recall |
|---|---|---|---|---|---|---|
| baseline | 0.775 | 0.778 | 0.148 | 1.000 | 0.500 | 1.000 |
| muril | 1.000 | 1.000 | 0.867 | 1.000 | 0.500 | 1.000 |

Test messages per language: en: 14, hi: 7, hinglish: 4.

The confidence interval comes from bootstrap resampling of the test set. When two models' intervals overlap heavily, the test set is too small to say which one is better.
