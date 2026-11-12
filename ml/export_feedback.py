"""Turn user feedback into labelled training rows.

Usage:
    python -m ml.export_feedback
    bash deploy/export-feedback.sh          # on the server (runs this inside Docker)

Each rated analysis gets a label from its feedback votes:

* "wrong" with a chosen label        -> that label
* "correct" on a Likely Scam verdict -> scam
* "correct" on a Safe verdict        -> legit, only with --confirmed-safe-as-legit,
                                        because the message could also be spam
* "correct" on Suspicious, or "wrong" without a label -> no label (skipped)

When several users rated the same analysis, the majority label wins; a tie is
skipped. Only anonymised text is exported.

Messages already saved in a reviewed file in data/labels/ are skipped, so each
export contains only new ratings. The file has an extra "keep" column set to
"yes": a reviewer changes it to "no" for rows that look wrong, then saves the
file in data/labels/, where ml.prepare_data picks it up for training.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

import app.models  # noqa: F401  (registers tables)
from app.core.config import BASE_DIR, get_settings
from app.core.database import build_engine, build_session_factory, utcnow
from app.models import Analysis, Feedback
from ml.dataset import (
    COLUMNS,
    KEEP_COLUMN,
    DatasetError,
    allowed_scam_types,
    reviewed_ids,
    validate_dataset,
)

RAW_DIR = BASE_DIR / "data" / "raw"
SOURCE = "user_feedback"


@dataclass
class ExportSummary:
    exported: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)


def implied_label(verdict: str, feedback: Feedback, confirmed_safe_as_legit: bool) -> str | None:
    if not feedback.is_correct:
        return feedback.correct_label
    if verdict == "likely_scam":
        return "scam"
    if verdict == "safe" and confirmed_safe_as_legit:
        return "legit"
    return None


def collect_rows(
    session: Session,
    *,
    since: datetime,
    min_votes: int = 1,
    confirmed_safe_as_legit: bool = False,
    skip_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[dict[str, str]], ExportSummary]:
    summary = ExportSummary()
    known_types = allowed_scam_types()
    analyses = session.scalars(
        select(Analysis)
        .join(Feedback)
        .where(Analysis.created_at >= since)
        .distinct()
        .order_by(Analysis.created_at)
    ).all()

    rows: list[dict[str, str]] = []
    for analysis in analyses:
        row_id = f"fb-{analysis.public_id}"
        if row_id in skip_ids:
            summary.skipped["already in a reviewed file"] += 1
            continue
        votes = Counter(
            label
            for fb in analysis.feedback
            if (label := implied_label(analysis.verdict, fb, confirmed_safe_as_legit))
        )
        if not votes:
            summary.skipped["no usable label"] += 1
            continue
        ranked = votes.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            summary.skipped["users disagreed"] += 1
            continue
        label, count = ranked[0]
        if count < min_votes:
            summary.skipped[f"fewer than {min_votes} votes"] += 1
            continue

        scam_type = ""
        if label == "scam":
            scam_type = analysis.scam_type if analysis.scam_type in known_types else "other"
        rows.append(
            {
                "id": row_id,
                "text": analysis.anonymized_text,
                "language": analysis.language,
                "label": label,
                "source": SOURCE,
                "scam_type": scam_type,
            }
        )
        summary.exported[label] += 1
    return rows, summary


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=RAW_DIR / f"feedback_{stamp}.csv")
    parser.add_argument("--database-url", default=None, help="defaults to DATABASE_URL")
    parser.add_argument("--days", type=int, default=365, help="only analyses from the last N days")
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="skip messages already saved in reviewed files here (default: data/labels/)",
    )
    parser.add_argument(
        "--confirmed-safe-as-legit",
        action="store_true",
        help="label Safe verdicts that users confirmed as legit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = build_engine(args.database_url or get_settings().database_url)
    try:
        with build_session_factory(engine)() as session:
            rows, summary = collect_rows(
                session,
                since=utcnow() - timedelta(days=args.days),
                min_votes=args.min_votes,
                confirmed_safe_as_legit=args.confirmed_safe_as_legit,
                skip_ids=reviewed_ids(args.labels_dir),
            )
    finally:
        engine.dispose()

    for reason, count in sorted(summary.skipped.items()):
        print(f"skipped {count}: {reason}")
    if not rows:
        print("No new feedback with a usable label; nothing written.")
        return 0

    frame = pd.DataFrame(rows, columns=COLUMNS)
    try:
        validate_dataset(frame, "feedback export")
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frame[KEEP_COLUMN] = "yes"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}: {dict(summary.exported)}")
    print("Review it (set keep to no for wrong rows), then save it in data/labels/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
