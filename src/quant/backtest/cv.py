"""Walk-forward purged cross-validation.

────────────────────────────────────────────────────────────────────────────
Why standard k-fold CV is wrong for time series
────────────────────────────────────────────────────────────────────────────
Standard k-fold randomly assigns rows to folds, so the train set contains
both rows BEFORE and AFTER the test rows. For time-series ML, this leaks
the future into training: the model implicitly learns "what comes next"
from neighboring future rows.

────────────────────────────────────────────────────────────────────────────
Walk-forward CV
────────────────────────────────────────────────────────────────────────────
Train on [t0, t1], test on [t1, t2], slide forward. At each fold, the
model only sees data that PRECEDES the test set. This mimics live trading
(you train on what you know, test on what comes next).

────────────────────────────────────────────────────────────────────────────
Purging
────────────────────────────────────────────────────────────────────────────
Even walk-forward leaks if your LABELS overlap the train/test boundary.
Suppose you predict 5-day forward returns. The label at row t depends on
prices from t to t+5. If the test set begins at t1, then training labels
at rows t1-5..t1-1 still depend on prices in the test period — leak.

Purging removes those rows from training. We delete a window of size
`label_horizon` from the END of training set, just before the test set.

────────────────────────────────────────────────────────────────────────────
Embargo
────────────────────────────────────────────────────────────────────────────
After the test set, we need to skip a window before the next training set
begins. Otherwise serial correlation in features (a price feature isn't
independent across consecutive bars) lets information from the test fold
leak forward into the next training set when folds are stacked.

The embargo is typically a few percent of the test window length.

────────────────────────────────────────────────────────────────────────────
Reference
────────────────────────────────────────────────────────────────────────────
Marcos López de Prado, "Advances in Financial Machine Learning" (2018),
Chapter 7. The single most important chapter in the book.

────────────────────────────────────────────────────────────────────────────
This module
────────────────────────────────────────────────────────────────────────────
We provide split *generators* — sequences of (train_start, train_end,
test_start, test_end) tuples — that consumers (backtester, ML training)
slice their data with. We don't slice data ourselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Split:
    """One CV fold."""

    train_start: datetime
    train_end: datetime  # exclusive
    test_start: datetime
    test_end: datetime  # exclusive

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days


def walk_forward_splits(
    *,
    start: datetime,
    end: datetime,
    train_window: timedelta,
    test_window: timedelta,
    step: timedelta | None = None,
    label_horizon: timedelta = timedelta(days=0),
    embargo: timedelta = timedelta(days=0),
    expanding: bool = False,
) -> Iterator[Split]:
    """Generate walk-forward splits with optional purge + embargo.

    Args:
        start, end: total date range (UTC, both inclusive boundaries OK).
        train_window: size of each training window. Ignored if expanding=True
            (in which case training always starts at `start`).
        test_window: size of each test window.
        step: how much to slide forward between folds. Default = test_window
            (non-overlapping test sets).
        label_horizon: time delta covered by labels. We purge this much from
            the end of training. Set to 0 if labels are 1-bar-ahead.
        embargo: time skipped after test set before next train set.
        expanding: if True, training window grows (anchored start, sliding end).

    Yields:
        Split objects.
    """
    step = step or test_window
    cur_train_end = start + train_window
    while True:
        test_start = cur_train_end + label_horizon  # purge
        test_end = test_start + test_window
        if test_end > end:
            break

        train_start = start if expanding else (cur_train_end - train_window)
        train_end = cur_train_end  # already excludes the purge window

        yield Split(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # Slide forward — embargo applies after the test set
        cur_train_end = test_end + embargo


def kfold_purged_splits(
    *,
    start: datetime,
    end: datetime,
    n_splits: int,
    label_horizon: timedelta = timedelta(days=0),
    embargo_pct: float = 0.01,
) -> Iterator[Split]:
    """K-fold splits with purging — alternative to walk-forward.

    Each fold uses one chunk as test, all earlier chunks as train.
    Like walk-forward but with equal-sized folds across the date range.

    Args:
        n_splits: number of test folds (so n_splits-1 train+test fold pairs)
        embargo_pct: embargo as fraction of total range (e.g. 0.01 = 1%)
    """
    total = end - start
    fold_size = total / n_splits
    embargo = timedelta(seconds=total.total_seconds() * embargo_pct)

    for i in range(1, n_splits):
        train_end = start + fold_size * i
        test_start = train_end + label_horizon
        test_end = test_start + fold_size
        if test_end > end:
            test_end = end
        yield Split(
            train_start=start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
