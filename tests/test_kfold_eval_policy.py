"""Tests for K-fold rotation evaluation policy."""

import pytest
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState
from gepa.strategies.eval_policy import KFoldRotationEvaluationPolicy


def test_kfold_initialization():
    """Test that K-fold policy initializes correctly."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)
    assert policy.num_folds == 3
    assert not policy._initialized


def test_kfold_invalid_num_folds():
    """Test that K-fold policy rejects invalid num_folds."""
    with pytest.raises(ValueError, match="num_folds must be at least 2"):
        KFoldRotationEvaluationPolicy(num_folds=1)


def test_kfold_partitions_data():
    """Test that K-fold policy partitions data into roughly equal folds."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)
    
    # Create a validation set with 10 items
    valset = [{"id": i, "split": "val"} for i in range(10)]
    loader = ListDataLoader(valset)
    
    # Initialize folds
    policy._initialize_folds(loader)
    
    # Check that all folds are created
    assert len(policy._folds) == 3
    
    # Check that all items are assigned to exactly one fold
    all_fold_ids = []
    for fold in policy._folds:
        all_fold_ids.extend(fold)
    assert sorted(all_fold_ids) == list(range(10))
    
    # Check fold sizes are roughly equal (difference of at most 1)
    fold_sizes = [len(fold) for fold in policy._folds]
    assert max(fold_sizes) - min(fold_sizes) <= 1


def test_kfold_rotation_through_iterations():
    """Test that K-fold policy rotates through folds across iterations."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)
    
    # Create a validation set with 9 items (divisible by 3 for clean folds)
    valset = [{"id": i, "split": "val"} for i in range(9)]
    loader = ListDataLoader(valset)
    
    # Create a mock state with increasing number of programs
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    
    # Iteration 0: fold 0 for selection, folds 1+2 for evaluation
    state.prog_candidate_val_subscores = [{}]  # 1 program = iteration 0
    eval_ids_0 = policy.get_eval_batch(loader, state)
    
    # Iteration 1: fold 1 for selection, folds 0+2 for evaluation
    state.prog_candidate_val_subscores = [{}, {}]  # 2 programs = iteration 1
    eval_ids_1 = policy.get_eval_batch(loader, state)
    
    # Iteration 2: fold 2 for selection, folds 0+1 for evaluation
    state.prog_candidate_val_subscores = [{}, {}, {}]  # 3 programs = iteration 2
    eval_ids_2 = policy.get_eval_batch(loader, state)
    
    # Iteration 3: back to fold 0 for selection, folds 1+2 for evaluation
    state.prog_candidate_val_subscores = [{}, {}, {}, {}]  # 4 programs = iteration 3
    eval_ids_3 = policy.get_eval_batch(loader, state)
    
    # Each iteration should evaluate 2/3 of the data (6 out of 9)
    assert len(eval_ids_0) == 6
    assert len(eval_ids_1) == 6
    assert len(eval_ids_2) == 6
    assert len(eval_ids_3) == 6
    
    # The evaluation sets should be different across the first 3 iterations
    assert set(eval_ids_0) != set(eval_ids_1)
    assert set(eval_ids_1) != set(eval_ids_2)
    assert set(eval_ids_0) != set(eval_ids_2)
    
    # Iteration 3 should match iteration 0 (cycle repeats)
    assert set(eval_ids_3) == set(eval_ids_0)
    
    # The complement of each eval set should be exactly one fold
    all_ids = set(range(9))
    assert len(all_ids - set(eval_ids_0)) == 3  # Selection fold size
    assert len(all_ids - set(eval_ids_1)) == 3
    assert len(all_ids - set(eval_ids_2)) == 3


def test_kfold_get_best_program():
    """Test that get_best_program works correctly."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)
    
    # Create state with multiple programs and scores
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [
        {0: 0.5, 1: 0.6},  # Program 0: avg = 0.55
        {0: 0.7, 1: 0.8},  # Program 1: avg = 0.75 (best)
        {0: 0.4, 1: 0.5},  # Program 2: avg = 0.45
    ]
    
    best_idx = policy.get_best_program(state)
    assert best_idx == 1


def test_kfold_get_valset_score():
    """Test that get_valset_score returns the average score."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)
    
    # Create state with a program that has scores
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [
        {0: 0.6, 1: 0.8, 2: 0.7},  # Program 0: avg = 0.7
    ]
    
    score = policy.get_valset_score(0, state)
    assert abs(score - 0.7) < 1e-6


def test_kfold_empty_valset():
    """Test that K-fold policy handles empty validation sets."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)
    
    loader = ListDataLoader([])
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [{}]
    
    eval_ids = policy.get_eval_batch(loader, state)
    assert eval_ids == []


def test_kfold_with_uneven_folds():
    """Test that K-fold policy handles validation sets not divisible by K."""
    policy = KFoldRotationEvaluationPolicy(num_folds=4)
    
    # 10 items, 4 folds -> folds of size 3, 3, 2, 2
    valset = [{"id": i, "split": "val"} for i in range(10)]
    loader = ListDataLoader(valset)
    
    policy._initialize_folds(loader)
    
    # Verify all items are in exactly one fold
    all_fold_ids = []
    for fold in policy._folds:
        all_fold_ids.extend(fold)
    assert sorted(all_fold_ids) == list(range(10))
    
    # Check that fold sizes differ by at most 1
    fold_sizes = [len(fold) for fold in policy._folds]
    assert max(fold_sizes) - min(fold_sizes) <= 1
    assert sum(fold_sizes) == 10
