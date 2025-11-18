"""Tests for RandomSplitEvaluationPolicy."""

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.eval_policy import RandomSplitEvaluationPolicy


class MockState:
    """Mock GEPA state for testing."""

    def __init__(self):
        self.prog_candidate_val_subscores = []


def test_random_split_initialization():
    """Test RandomSplitEvaluationPolicy initialization."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    assert policy.selection_ratio == 0.5
    
    # Test invalid ratio
    with pytest.raises(ValueError, match="selection_ratio must be between 0 and 1"):
        RandomSplitEvaluationPolicy(selection_ratio=0)
    
    with pytest.raises(ValueError, match="selection_ratio must be between 0 and 1"):
        RandomSplitEvaluationPolicy(selection_ratio=1.0)
    
    with pytest.raises(ValueError, match="selection_ratio must be between 0 and 1"):
        RandomSplitEvaluationPolicy(selection_ratio=1.5)


def test_random_split_batch_selection():
    """Test that get_eval_batch returns the selection subset."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(10)])
    state = MockState()
    
    # Get the selection batch
    batch = policy.get_eval_batch(loader, state)
    
    # Should get approximately 50% of the data
    assert len(batch) == 5
    
    # Should return the same batch on subsequent calls (persistent split)
    batch2 = policy.get_eval_batch(loader, state)
    assert batch == batch2


def test_random_split_different_ratios():
    """Test different selection ratios."""
    loader = ListDataLoader([{"id": i} for i in range(100)])
    state = MockState()
    
    # Test 30% selection
    policy_30 = RandomSplitEvaluationPolicy(selection_ratio=0.3, seed=42)
    batch_30 = policy_30.get_eval_batch(loader, state)
    assert len(batch_30) == 30
    
    # Test 70% selection
    policy_70 = RandomSplitEvaluationPolicy(selection_ratio=0.7, seed=42)
    batch_70 = policy_70.get_eval_batch(loader, state)
    assert len(batch_70) == 70


def test_random_split_empty_loader():
    """Test behavior with empty data loader."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    loader = ListDataLoader([])
    state = MockState()
    
    batch = policy.get_eval_batch(loader, state)
    assert batch == []


def test_random_split_seed_reproducibility():
    """Test that same seed produces same split."""
    loader = ListDataLoader([{"id": i} for i in range(10)])
    state1 = MockState()
    state2 = MockState()
    
    policy1 = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    policy2 = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    
    batch1 = policy1.get_eval_batch(loader, state1)
    batch2 = policy2.get_eval_batch(loader, state2)
    
    assert batch1 == batch2


def test_centered_scoring():
    """Test that get_best_program uses centered scores."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    state = MockState()
    
    # Create scores where programs perform differently on different tasks
    # Task 0: easy (all programs score high)
    # Task 1: hard (all programs score low)
    # Task 2: medium
    state.prog_candidate_val_subscores = [
        {0: 0.9, 1: 0.1, 2: 0.5},  # Program 0: consistent
        {0: 0.8, 1: 0.2, 2: 0.6},  # Program 1: slightly better on hard task
        {0: 1.0, 1: 0.0, 2: 0.4},  # Program 2: best on easy, worst on hard
    ]
    
    # Task means: 0: 0.9, 1: 0.1, 2: 0.5
    # Centered scores:
    # Program 0: [0.0, 0.0, 0.0] -> avg: 0.0
    # Program 1: [-0.1, 0.1, 0.1] -> avg: 0.033...
    # Program 2: [0.1, -0.1, -0.1] -> avg: -0.033...
    
    best_idx = policy.get_best_program(state)
    assert best_idx == 1  # Program 1 has highest centered score


def test_centered_scoring_with_coverage():
    """Test that coverage is used as tiebreaker."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    state = MockState()
    
    # Programs with same centered average but different coverage
    state.prog_candidate_val_subscores = [
        {0: 1.0},  # Program 0: one task, centered score 0
        {0: 1.0, 1: 1.0},  # Program 1: two tasks, centered score 0
    ]
    
    best_idx = policy.get_best_program(state)
    assert best_idx == 1  # Program 1 has same centered score but more coverage


def test_get_valset_score():
    """Test get_valset_score returns centered score."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    state = MockState()
    
    state.prog_candidate_val_subscores = [
        {0: 1.0, 1: 0.5},
        {0: 0.5, 1: 1.0},
    ]
    
    # Task means: 0: 0.75, 1: 0.75
    # Program 0 centered: [0.25, -0.25] -> avg: 0.0
    score = policy.get_valset_score(0, state)
    assert abs(score - 0.0) < 1e-6
    
    # Invalid index
    score_invalid = policy.get_valset_score(-1, state)
    assert score_invalid == float("-inf")


def test_get_best_program_empty_state():
    """Test get_best_program with empty state."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    state = MockState()
    
    best_idx = policy.get_best_program(state)
    assert best_idx == -1


def test_get_best_program_no_scores():
    """Test get_best_program when programs have no scores."""
    policy = RandomSplitEvaluationPolicy(selection_ratio=0.5, seed=42)
    state = MockState()
    
    state.prog_candidate_val_subscores = [{}, {}, {}]
    
    best_idx = policy.get_best_program(state)
    assert best_idx == -1
