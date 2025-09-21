"""
Dataset interfaces and utilities for probelib.

This module provides the core dataset abstractions for working with dialogue data,
including mask-based token selection and metadata handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from ..masks import MaskFunction, assistant
from ..types import Dialogue, DialogueDataType, Label


class DialogueDataset(ABC):
    """
    Abstract base class for dialogue datasets.

    This class provides the core interface for working with dialogue data in probelib.

    Key features:
    - Mask-based token selection
    - Metadata handling
    - Label management
    - Dialogue shuffling and slicing
    """

    # Class attributes to be set by subclasses
    base_name: str = "base_dataset"

    @property
    def default_mask(self) -> MaskFunction:
        """
        Default mask for this dataset.

        By default, uses assistant-only mask (most common case).
        Subclasses can override this for different behavior.
        """
        return assistant()

    def __init__(
        self,
        dialogues: list[Dialogue] | None = None,
        labels: list[Label] | None = None,
        metadata: dict[str, list[Any]] | None = None,
        shuffle_upon_init: bool = True,
        shuffle_seed: int | None = 42,
        **kwargs,
    ):
        """
        Initialize the DialogueDataset.

        Args:
            dialogues: List of dialogues (if None, will call _get_dialogues)
            labels: List of labels corresponding to dialogues
            metadata: Optional metadata dictionary
            shuffle_upon_init: Whether to shuffle dialogues on initialization
        """

        if dialogues is None:
            assert labels is None and metadata is None, (
                "If dialogues is None, labels and metadata must also be None"
            )
            dialogues, labels, metadata = self._get_dialogues(**kwargs)

        assert labels is not None, "Labels must be provided"

        self.dialogues = dialogues
        self.labels = labels
        self.metadata = metadata
        self._shuffle_seed = shuffle_seed
        self._shuffle_rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random.default_rng()
        )

        self.post_process()

        # Shuffle for better train/test splits
        if len(set(self.labels)) > 1 and shuffle_upon_init:
            self.randomly_shuffle_dialogues()

    @abstractmethod
    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        """
        Abstract method to load dialogues, labels, and metadata.

        Must be implemented by subclasses to define how data is loaded.

        Returns:
            Tuple of (dialogues, labels, metadata)
        """
        pass

    def post_process(self) -> None:
        """Post-process the dataset. Override in subclasses if needed."""
        pass

    def randomly_shuffle_dialogues(self) -> None:
        """Randomly shuffle dialogues, labels, and metadata."""
        perm = self._shuffle_rng.permutation(len(self))

        self.dialogues = [self.dialogues[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]

        if self.metadata is not None:
            self.metadata = {k: [v[i] for i in perm] for k, v in self.metadata.items()}

    def _slice_by_index(self, indices: int | list[int]) -> DialogueDataType:
        """Slice dataset by indices."""
        if isinstance(indices, int):
            indices = [indices]

        dialogues = [self.dialogues[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        metadata = None

        if self.metadata is not None:
            metadata = {k: [v[i] for i in indices] for k, v in self.metadata.items()}

        return dialogues, labels, metadata

    def subset_where_true(self, condition: list[bool]) -> Self:
        """Filter dataset by boolean condition."""
        assert len(condition) == len(self)
        indices = [i for i, c in enumerate(condition) if c]

        dialogues, labels, metadata = self._slice_by_index(indices)

        new_dataset = self.__class__(
            dialogues=dialogues,
            labels=labels,
            metadata=metadata,
            shuffle_upon_init=False,
            shuffle_seed=self._shuffle_seed,
        )
        new_dataset.base_name = self.base_name

        return new_dataset

    def get_with_label(self, label: Label) -> Self:
        """Get subset with specific label."""
        return self.subset_where_true([_label == label for _label in self.labels])

    def get_negative(self) -> Self:
        """Get negative examples."""
        return self.get_with_label(Label.NEGATIVE)

    def get_positive(self) -> Self:
        """Get positive examples."""
        return self.get_with_label(Label.POSITIVE)

    def __len__(self) -> int:
        """Return number of dialogues."""
        return len(self.dialogues)

    def __getitem__(self, idx: int | slice | list) -> Self:
        """Get dialogue(s) by index/slice/list."""
        if isinstance(idx, int):
            dialogues, labels, metadata = self._slice_by_index(idx)
        elif isinstance(idx, slice):
            indices = list(range(len(self)))[idx]
            dialogues, labels, metadata = self._slice_by_index(indices)
        elif isinstance(idx, list):
            dialogues, labels, metadata = self._slice_by_index(idx)
        else:
            raise TypeError(
                f"Dataset indices must be integers, slices, or lists, not {type(idx).__name__}"
            )

        new_dataset = self.__class__(
            dialogues=dialogues,
            labels=labels,
            metadata=metadata,
            shuffle_upon_init=False,
            shuffle_seed=self._shuffle_seed,
        )
        new_dataset.base_name = self.base_name

        return new_dataset

    def __add__(self, other: "DialogueDataset") -> "DialogueDataset":
        """Concatenate two datasets."""
        dialogues = self.dialogues + other.dialogues
        labels = self.labels + other.labels

        if self.metadata is not None and other.metadata is not None:
            assert set(self.metadata.keys()) == set(other.metadata.keys()), (
                "Metadata key mismatch"
            )
            metadata = {k: self.metadata[k] + other.metadata[k] for k in self.metadata}
        else:
            metadata = None

        new_base_name = f"{self.base_name}+{other.base_name}"

        new_dataset = self.__class__(
            dialogues=dialogues,
            labels=labels,
            metadata=metadata,
            shuffle_upon_init=True,
            shuffle_seed=self._shuffle_seed,
        )
        new_dataset.base_name = new_base_name

        return new_dataset

    def split(self, proportion: float = 0.8, shuffle: bool = True) -> tuple[Self, Self]:
        """
        Split dataset into two parts.

        Args:
            proportion: Fraction of data for first split (between 0 and 1)
            shuffle: Whether to shuffle before splitting (passed to shuffle_upon_init)

        Returns:
            Tuple of (first_split, second_split) datasets

        Example:
            >>> dataset = DolusChatDataset()
            >>> train, test = dataset.split(0.8)
            >>> train, test = dataset.split(0.8, shuffle=False)
        """
        if not 0 < proportion < 1:
            raise ValueError(f"Proportion must be between 0 and 1, got {proportion}")

        split_idx = int(len(self) * proportion)

        first_dialogues = self.dialogues[:split_idx]
        first_labels = self.labels[:split_idx]
        first_metadata = None
        if self.metadata is not None:
            first_metadata = {k: v[:split_idx] for k, v in self.metadata.items()}

        second_dialogues = self.dialogues[split_idx:]
        second_labels = self.labels[split_idx:]
        second_metadata = None
        if self.metadata is not None:
            second_metadata = {k: v[split_idx:] for k, v in self.metadata.items()}

        first_split = self.__class__(
            dialogues=first_dialogues,
            labels=first_labels,
            metadata=first_metadata,
            shuffle_upon_init=shuffle,
            shuffle_seed=self._shuffle_seed,
        )
        first_split.base_name = self.base_name

        second_split = self.__class__(
            dialogues=second_dialogues,
            labels=second_labels,
            metadata=second_metadata,
            shuffle_upon_init=shuffle,
            shuffle_seed=self._shuffle_seed,
        )
        second_split.base_name = self.base_name

        return first_split, second_split

    @property
    def name(self) -> str:
        """Get dataset name."""
        return self.base_name
