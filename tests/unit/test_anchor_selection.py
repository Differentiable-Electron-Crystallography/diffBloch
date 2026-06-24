"""Quartz anchor rotation selection supports fast defaults and explicit subsets."""

import pytest

from diffBloch.config import select_reference_rotations

ROTATIONS = [{"rotation_idx": 26}, {"rotation_idx": 33}, {"rotation_idx": 56}]


def test_select_first_n_rotations() -> None:
    assert select_reference_rotations(ROTATIONS, "first:2") == ROTATIONS[:2]


def test_select_all_rotations() -> None:
    assert select_reference_rotations(ROTATIONS, "all") == ROTATIONS


def test_select_explicit_rotation_indices() -> None:
    assert select_reference_rotations(ROTATIONS, "56,26") == [ROTATIONS[0], ROTATIONS[2]]


def test_select_rejects_missing_rotation_indices() -> None:
    with pytest.raises(ValueError, match="unknown rotation_idx"):
        select_reference_rotations(ROTATIONS, "999")


def test_select_rejects_empty_selector() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        select_reference_rotations(ROTATIONS, "")
    with pytest.raises(ValueError, match="must not be empty"):
        select_reference_rotations(ROTATIONS, "  ")
