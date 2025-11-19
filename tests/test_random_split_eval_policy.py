"""Tests for RandomSplitEvaluationPolicy."""

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.eval_policy import RandomSplitEvaluationPolicy


class MockState:
    """Mock GEPA state for testing."""

    def __init__(self):
        self.prog_candidate_val_subscores = []
    
    def get_program_average_val_subset(self, program_idx):
        """Mock implementation of get_program_average_val_subset."""
        if program_idx < 0 or program_idx >= len(self.prog_candidate_val_subscores):
            return float("-inf"), 0
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        avg = sum(scores.values()) / len(scores)
        return avg, len(scores)


def test_random_split_initialization():
    """Test RandomSplitEvaluationPolicy initialization."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    assert policy.evaluation_ratio == 0.5
    
    # Test invalid ratio
    with pytest.raises(ValueError, match="evaluation_ratio must be between 0 and 1"):
        RandomSplitEvaluationPolicy(evaluation_ratio=0)
    
    with pytest.raises(ValueError, match="evaluation_ratio must be between 0 and 1"):
        RandomSplitEvaluationPolicy(evaluation_ratio=1.0)
    
    with pytest.raises(ValueError, match="evaluation_ratio must be between 0 and 1"):
        RandomSplitEvaluationPolicy(evaluation_ratio=1.5)


def test_random_split_batch_selection():
    """Test that get_eval_batch returns the evaluation subset."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(10)])
    state = MockState()
    
    # Get the evaluation batch
    batch = policy.get_eval_batch(loader, state)
    
    # Should get approximately 50% of the data (evaluation subset when evaluation_ratio=0.5)
    assert len(batch) == 5
    
    # Should return the same batch on subsequent calls (persistent split)
    batch2 = policy.get_eval_batch(loader, state)
    assert batch == batch2


def test_random_split_different_ratios():
    """Test different evaluation ratios."""
    loader = ListDataLoader([{"id": i} for i in range(100)])
    state = MockState()
    
    # Test 70% evaluation (30% selection)
    policy_70 = RandomSplitEvaluationPolicy(evaluation_ratio=0.7, seed=42)
    batch_70 = policy_70.get_eval_batch(loader, state)
    assert len(batch_70) == 70  # Returns evaluation batch, which is 70%
    
    # Test 30% evaluation (70% selection)
    policy_30 = RandomSplitEvaluationPolicy(evaluation_ratio=0.3, seed=42)
    batch_30 = policy_30.get_eval_batch(loader, state)
    assert len(batch_30) == 30  # Returns evaluation batch, which is 30%


def test_random_split_empty_loader():
    """Test behavior with empty data loader."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    loader = ListDataLoader([])
    state = MockState()
    
    batch = policy.get_eval_batch(loader, state)
    assert batch == []


def test_random_split_seed_reproducibility():
    """Test that same seed produces same split."""
    loader = ListDataLoader([{"id": i} for i in range(10)])
    state1 = MockState()
    state2 = MockState()
    
    policy1 = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    policy2 = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    
    batch1 = policy1.get_eval_batch(loader, state1)
    batch2 = policy2.get_eval_batch(loader, state2)
    
    assert batch1 == batch2


def test_advantage_scoring():
    """Test that get_best_program uses advantages."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
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
    # Advantages:
    # Program 0: [0.0, 0.0, 0.0] -> avg: 0.0
    # Program 1: [-0.1, 0.1, 0.1] -> avg: 0.033...
    # Program 2: [0.1, -0.1, -0.1] -> avg: -0.033...
    
    best_idx = policy.get_best_program(state)
    assert best_idx == 1  # Program 1 has highest advantage


def test_advantage_scoring_with_coverage():
    """Test that coverage is used as tiebreaker."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    state = MockState()
    
    # Programs with same advantage average but different coverage
    state.prog_candidate_val_subscores = [
        {0: 1.0},  # Program 0: one task, advantage 0
        {0: 1.0, 1: 1.0},  # Program 1: two tasks, advantage 0
    ]
    
    best_idx = policy.get_best_program(state)
    assert best_idx == 1  # Program 1 has same advantage but more coverage


def test_get_valset_score():
    """Test get_valset_score returns raw average score."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    state = MockState()
    
    state.prog_candidate_val_subscores = [
        {0: 1.0, 1: 0.5},
        {0: 0.5, 1: 1.0},
    ]
    
    # Test get_valset_score - should return raw average
    # Program 0: [1.0, 0.5] -> avg: 0.75
    score = policy.get_valset_score(0, state)
    assert abs(score - 0.75) < 1e-6
    
    # Test get_advantage - should return advantage
    # Task means: 0: 0.75, 1: 0.75
    # Program 0 advantage: [0.25, -0.25] -> avg: 0.0
    advantage = policy.get_advantage(0, state)
    assert abs(advantage - 0.0) < 1e-6
    
    # Invalid index for both methods
    score_invalid = policy.get_valset_score(-1, state)
    assert score_invalid == float("-inf")
    
    advantage_invalid = policy.get_advantage(-1, state)
    assert advantage_invalid == float("-inf")


def test_get_best_program_empty_state():
    """Test get_best_program with empty state."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    state = MockState()
    
    best_idx = policy.get_best_program(state)
    assert best_idx == -1


def test_get_best_program_no_scores():
    """Test get_best_program when programs have no scores."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    state = MockState()
    
    state.prog_candidate_val_subscores = [{}, {}, {}]
    
    best_idx = policy.get_best_program(state)
    assert best_idx == -1
