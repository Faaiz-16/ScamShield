"""Baseline model: TF-IDF features over words and characters + logistic regression.

Character n-grams make the model robust to Hinglish spelling variation
("jayega" / "jaega" / "jaayega") and to Devanagari, where word tokenisation is
less reliable. Word n-grams capture phrases such as "share otp".
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from app.detection.preprocessing import prepare_batch

# Placeholders such as "<phone>" stay whole; otherwise split on whitespace and
# punctuation. The default pattern (\w\w+) would cut Hindi words at vowel signs.
TOKEN_PATTERN = r"<\w+>|[^\s.,!?;:()\[\]\"'।|/\\-]+"


def build_pipeline(c: float = 4.0, seed: int = 42) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=(1, 2),
                    lowercase=False,  # prepare_batch already case-folds
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    lowercase=False,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed)
    return Pipeline(
        [
            ("prepare", FunctionTransformer(prepare_batch)),
            ("features", features),
            ("classifier", classifier),
        ]
    )
