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

    # Iteration 0: fold 0 for evaluation, folds 1+2 for selection
    state.prog_candidate_val_subscores = [{}]  # 1 program = iteration 0
    eval_ids_0 = policy.get_eval_batch(loader, state)

    # Iteration 1: fold 1 for evaluation, folds 0+2 for selection
    state.prog_candidate_val_subscores = [{}, {}]  # 2 programs = iteration 1
    eval_ids_1 = policy.get_eval_batch(loader, state)

    # Iteration 2: fold 2 for evaluation, folds 0+1 for selection
    state.prog_candidate_val_subscores = [{}, {}, {}]  # 3 programs = iteration 2
    eval_ids_2 = policy.get_eval_batch(loader, state)

    # Iteration 3: back to fold 0 for evaluation, folds 1+2 for selection
    state.prog_candidate_val_subscores = [{}, {}, {}, {}]  # 4 programs = iteration 3
    eval_ids_3 = policy.get_eval_batch(loader, state)

    # Each iteration should evaluate 1/3 of the data (3 out of 9)
    assert len(eval_ids_0) == 3
    assert len(eval_ids_1) == 3
    assert len(eval_ids_2) == 3
    assert len(eval_ids_3) == 3

    # The evaluation sets should be different across the first 3 iterations
    assert set(eval_ids_0) != set(eval_ids_1)
    assert set(eval_ids_1) != set(eval_ids_2)
    assert set(eval_ids_0) != set(eval_ids_2)

    # Iteration 3 should match iteration 0 (cycle repeats)
    assert set(eval_ids_3) == set(eval_ids_0)

    # Each eval set should be exactly one fold
    all_ids = set(range(9))
    # The union of all three unique eval sets should be all ids
    all_eval_ids = set(eval_ids_0) | set(eval_ids_1) | set(eval_ids_2)
    assert all_eval_ids == all_ids


def test_kfold_get_best_program():
    """Test that get_best_program only considers candidate selection folds."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)

    # Create a validation set with 9 items (divisible by 3 for clean folds)
    valset = [{"id": i, "split": "val"} for i in range(9)]
    loader = ListDataLoader(valset)

    # Initialize the policy by calling get_eval_batch
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [{}]  # 1 program = len=1, 1%3=1, fold 1 is eval
    policy.get_eval_batch(loader, state)  # This initializes folds

    # Now set up test with 3 programs (len=3, 3%3=0, fold 0 is eval)
    # Fold 0 (ids 0,1,2) is for evaluation
    # Folds 1,2 (ids 3,4,5,6,7,8) are for candidate selection
    state.prog_candidate_val_subscores = [
        # Program 0: high on eval fold, low on selection folds
        {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.3, 4: 0.3, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.3},
        # Program 1: low on eval fold, high on selection folds (should be best)
        {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.9, 7: 0.9, 8: 0.9},
        # Program 2: medium scores everywhere
        {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5},
    ]

    best_idx = policy.get_best_program(state)
    # Program 1 should be best because it has highest avg on candidate selection folds
    # even though it has lowest score on evaluation fold
    assert best_idx == 1


def test_kfold_get_valset_score():
    """Test that get_valset_score only considers candidate selection folds."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)

    # Create a validation set with 9 items
    valset = [{"id": i, "split": "val"} for i in range(9)]
    loader = ListDataLoader(valset)

    # Initialize the policy with 1 program (iteration 0/seed)
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [{}]
    policy.get_eval_batch(loader, state)  # Initialize folds

    # At iteration with 1 program: len=1, 1%3=1, so fold 1 (ids 3,4,5) is for evaluation
    # Folds 0,2 (ids 0,1,2,6,7,8) are for candidate selection
    state.prog_candidate_val_subscores = [
        # High score on eval fold 1 (should be ignored), low on selection folds 0,2
        {0: 0.2, 1: 0.2, 2: 0.2, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.2, 7: 0.2, 8: 0.2},
    ]

    score = policy.get_valset_score(0, state)
    # Should be avg of selection folds only (0,2): (0.2 * 6) / 6 = 0.2
    assert abs(score - 0.2) < 1e-6


def test_kfold_empty_valset():
    """Test that K-fold policy handles empty validation sets."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)

    loader = ListDataLoader([])
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [{}]

    eval_ids = policy.get_eval_batch(loader, state)
    assert eval_ids == []


def test_kfold_candidate_selection_excludes_eval_fold():
    """Test that candidate selection uses all folds except the evaluation fold."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)

    # Create a validation set with 9 items (3 per fold)
    valset = [{"id": i, "split": "val"} for i in range(9)]
    loader = ListDataLoader(valset)

    # Initialize state at iteration 0
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [{}]

    # Get eval fold for iteration 0 (should be fold 0: ids 0,1,2)
    eval_ids = policy.get_eval_batch(loader, state)
    assert len(eval_ids) == 3
    eval_set = set(eval_ids)

    # Get candidate selection ids (should be folds 1,2: ids 3,4,5,6,7,8)
    selection_ids = policy._get_candidate_selection_ids(state)
    assert len(selection_ids) == 6

    # Verify eval and selection sets are disjoint
    assert eval_set.isdisjoint(selection_ids)

    # Verify they cover all ids
    assert eval_set.union(selection_ids) == set(range(9))


def test_kfold_final_methods_use_all_scores():
    """Test that final methods use all available validation scores, not just selection folds."""
    policy = KFoldRotationEvaluationPolicy(num_folds=3)

    # Create a validation set with 9 items
    valset = [{"id": i, "split": "val"} for i in range(9)]
    loader = ListDataLoader(valset)

    # Initialize the policy with 1 program
    state = GEPAState(seed_candidate={}, base_valset_eval_output=({}, {}))
    state.prog_candidate_val_subscores = [{}]
    policy.get_eval_batch(loader, state)  # Initialize folds

    # With 2 programs: len=2, 2%3=2, so fold 2 (ids 6,7,8) is for evaluation
    # Folds 0,1 (ids 0,1,2,3,4,5) are for candidate selection
    # Set up programs where selection-only and full scores differ
    state.prog_candidate_val_subscores = [
        # Program 0: low on selection folds (0,1), high on eval fold (2)
        {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 1.0, 7: 1.0, 8: 1.0},
        # Program 1: high on selection folds (0,1), low on eval fold (2)
        {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.1, 7: 0.1, 8: 0.1},
    ]

    # During optimization: use selection folds only (folds 0,1)
    best_idx_during = policy.get_best_program(state)
    score_0_during = policy.get_valset_score(0, state)
    score_1_during = policy.get_valset_score(1, state)

    # Program 1 should be best during optimization (1.0 avg on selection folds vs 0.1)
    assert best_idx_during == 1
    assert abs(score_0_during - 0.1) < 1e-6  # Avg of selection folds for program 0
    assert abs(score_1_during - 1.0) < 1e-6  # Avg of selection folds for program 1

    # At the end: use all scores
    best_idx_final = policy.get_best_program_final(state)
    score_0_final = policy.get_valset_score_final(0, state)
    score_1_final = policy.get_valset_score_final(1, state)

    # Overall averages:
    # Program 0: (6*0.1 + 3*1.0) / 9 = 3.6/9 = 0.4
    # Program 1: (6*1.0 + 3*0.1) / 9 = 6.3/9 = 0.7
    # So program 1 is still best, but scores are different
    assert best_idx_final == 1
    assert abs(score_0_final - 0.4) < 1e-6
    assert abs(score_1_final - 0.7) < 1e-6

    # Verify the final scores are different from selection-only scores
    assert abs(score_0_final - score_0_during) > 0.2  # 0.4 vs 0.1
    assert abs(score_1_final - score_1_during) > 0.2  # 0.7 vs 1.0


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
