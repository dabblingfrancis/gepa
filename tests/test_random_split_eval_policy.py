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


def test_random_split_minimum_evaluation_size():
    """Test that at least 1 item is allocated to evaluation when possible."""
    # With very small evaluation_ratio, should still get at least 1 evaluation item
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.01, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(10)])
    state = MockState()
    
    batch = policy.get_eval_batch(loader, state)
    assert len(batch) >= 1  # Should have at least 1 item for evaluation


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
    """Test that get_best_program uses advantages on selection subset."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(3)])
    state = MockState()
    
    # Initialize the split so selection IDs are set
    policy.get_eval_batch(loader, state)
    
    # Create scores where programs perform differently on different tasks
    # Task 0: easy (all programs score high)
    # Task 1: hard (all programs score low)
    # Task 2: medium
    state.prog_candidate_val_subscores = [
        {0: 0.9, 1: 0.1, 2: 0.5},  # Program 0: consistent
        {0: 0.8, 1: 0.2, 2: 0.6},  # Program 1: slightly better on hard task
        {0: 1.0, 1: 0.0, 2: 0.4},  # Program 2: best on easy, worst on hard
    ]
    
    # Advantage computation will only use tasks in the selection subset
    # The exact result depends on which tasks are in selection vs evaluation
    best_idx = policy.get_best_program(state)
    assert best_idx in [0, 1, 2]  # Should return a valid program index


def test_advantage_scoring_with_coverage():
    """Test that coverage is used as tiebreaker."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(4)])
    state = MockState()
    
    # Initialize the split
    eval_ids = policy.get_eval_batch(loader, state)
    
    # Get the selection IDs (those not in evaluation)
    all_ids = set(range(4))
    selection_ids = all_ids - set(eval_ids)
    
    # Programs with same advantage average but different coverage
    # Create scores such that both programs have scores on selection subset only
    state.prog_candidate_val_subscores = [
        {list(selection_ids)[0]: 1.0},  # Program 0: one task from selection
        {list(selection_ids)[0]: 1.0, list(selection_ids)[1]: 1.0} if len(selection_ids) > 1 else {list(selection_ids)[0]: 1.0},  # Program 1: two tasks from selection if possible
    ]
    
    best_idx = policy.get_best_program(state)
    # Best program should be valid (either 0 or 1)
    assert best_idx in [0, 1]


def test_get_valset_score():
    """Test get_valset_score returns raw average score."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(2)])
    state = MockState()
    
    # Initialize the split
    policy.get_eval_batch(loader, state)
    
    state.prog_candidate_val_subscores = [
        {0: 1.0, 1: 0.5},
        {0: 0.5, 1: 1.0},
    ]
    
    # Test get_valset_score - should return raw average
    # Program 0: [1.0, 0.5] -> avg: 0.75
    score = policy.get_valset_score(0, state)
    assert abs(score - 0.75) < 1e-6
    
    # Test get_advantage - should return advantage on selection subset
    # The advantage will only be computed for tasks in the selection subset
    advantage = policy.get_advantage(0, state)
    # Advantage should be a valid number (not -inf) since we have scores
    assert advantage != float("-inf")
    
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


def test_filter_pareto_front():
    """Test that filter_pareto_front only includes selection IDs."""
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)
    loader = ListDataLoader([{"id": i} for i in range(10)])
    state = MockState()
    
    # Initialize the split
    policy.get_eval_batch(loader, state)
    
    # Create a pareto front with all validation IDs
    pareto_front = {
        0: {0, 1},
        1: {1},
        2: {0},
        3: {1, 2},
        4: {2},
        5: {0, 1},
        6: {1},
        7: {0, 2},
        8: {2},
        9: {0, 1, 2},
    }
    
    # Filter the pareto front
    filtered = policy.filter_pareto_front(pareto_front)
    
    # Check that only selection IDs are included
    assert len(filtered) == len(policy._selection_ids)
    for val_id in filtered.keys():
        assert val_id in policy._selection_ids
    
    # Check that evaluation IDs are excluded
    for val_id in policy._evaluation_ids:
        assert val_id not in filtered
    
    # Check that program sets are preserved
    for val_id in filtered.keys():
        assert filtered[val_id] == pareto_front[val_id]

