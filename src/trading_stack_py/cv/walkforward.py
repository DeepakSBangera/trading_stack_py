from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardCV:
    train_size: int
    test_size: int
    step_size: int | None = None
    expanding: bool = False
    min_train_size: int | None = None
    embargo: int = 0  # new: cut training right before test by this many obs

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.step_size is None:
            self.step_size = self.test_size
        if self.min_train_size is None:
            self.min_train_size = self.train_size

        start_train = 0
        end_train = self.train_size
        start_test = end_train
        end_test = start_test + self.test_size

        while end_test <= n_samples:
            # Apply embargo: ensure train ends before (start_test - embargo)
            effective_end_train = end_train
            if self.embargo > 0:
                effective_end_train = min(effective_end_train, max(0, start_test - self.embargo))

            train_idx = np.arange(0 if self.expanding else start_train, effective_end_train)
            if len(train_idx) < self.min_train_size:
                break

            test_idx = np.arange(start_test, end_test)
            yield train_idx, test_idx

            # advance window
            if self.expanding:
                end_train = end_train + self.step_size
            else:
                start_train = start_train + self.step_size
                end_train = end_train + self.step_size
            start_test = start_test + self.step_size
            end_test = start_test + self.test_size

    def run_fit_predict(
        self, estimator, X: np.ndarray, y: np.ndarray | None = None
    ) -> Iterable[np.ndarray]:
        preds = []
        for tr, te in self.split(len(X)):
            est = estimator()
            est.fit(X[tr], None if y is None else y[tr])
            p = est.predict(X[te])
            preds.append(p)
        return preds
