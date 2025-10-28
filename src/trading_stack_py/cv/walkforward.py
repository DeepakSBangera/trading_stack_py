from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardCV:
    train_size: int
    test_size: int
    step_size: int | None = None
    expanding: bool = False
    embargo: int = 0
    min_train_size: int | None = None

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        step = self.test_size if self.step_size is None else self.step_size
        emb = max(0, self.embargo)

        # Minimum train must reflect the trimmed tail
        default_min = max(1, self.train_size - emb)
        min_train = (
            default_min if self.min_train_size is None else max(1, self.min_train_size)
        )

        start_train = 0
        end_train = self.train_size  # nominal train end before trimming

        while True:
            # Trim the last 'emb' points from the train for fitting
            effective_end_train = max(start_train, end_train - emb)
            train_idx = np.arange(start_train, effective_end_train, dtype=int)

            # Test start: gap for non-expanding; no shift for expanding
            start_test = end_train + (0 if self.expanding else emb)
            end_test = start_test + self.test_size

            if end_test > n_samples:
                break

            if train_idx.size >= min_train:
                test_idx = np.arange(start_test, end_test, dtype=int)
                yield train_idx, test_idx

            # advance windows
            if self.expanding:
                end_train += step
            else:
                start_train += step
                end_train += step
