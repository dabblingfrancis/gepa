import importlib.util
import sys
import types
from pathlib import Path
import typing as t

import pytest

# Create minimal stub modules to satisfy imports inside eval_policy without importing the full package.
pkg_name = "gepa"
core_name = "gepa.core"
data_loader_name = "gepa.core.data_loader"
state_name = "gepa.core.state"

# gepa package stub
gepa_mod = types.ModuleType(pkg_name)
gepa_mod.__path__ = []  # mark as package
sys.modules[pkg_name] = gepa_mod

# gepa.core stub
core_mod = types.ModuleType(core_name)
core_mod.__path__ = []
sys.modules[core_name] = core_mod

# gepa.core.data_loader stub
dl_mod = types.ModuleType(data_loader_name)
# Provide TypeVars (Protocol[...] requires type variables)
dl_mod.DataId = t.TypeVar("DataId")
dl_mod.DataInst = t.TypeVar("DataInst")

class DataLoader:
    def __init__(self, ids):
        self._ids = list(ids)

    def all_ids(self):
        return list(self._ids)

dl_mod.DataLoader = DataLoader
sys.modules[data_loader_name] = dl_mod

# gepa.core.state stub
state_mod = types.ModuleType(state_name)
state_mod.ProgramIdx = int

class GEPAState:
    def __init__(self, i=0, prog_candidate_val_subscores=None, avg_map=None):
        self.i = i
        self.prog_candidate_val_subscores = [] if prog_candidate_val_subscores is None else prog_candidate_val_subscores
        self._avg_map = {} if avg_map is None else avg_map

    def get_program_average_val_subset(self, program_idx):
        return self._avg_map.get(program_idx, (float("-inf"), 0))

state_mod.GEPAState = GEPAState
sys.modules[state_name] = state_mod

# Load the eval_policy module directly from src path to avoid importing gepa package __init__.
repo_root = Path(__file__).resolve().parents[1]
eval_policy_path = repo_root / "src" / "gepa" / "strategies" / "eval_policy.py"
spec = importlib.util.spec_from_file_location("eval_policy_test_mod", str(eval_policy_path))
eval_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_policy)

RandomSplitEvaluationPolicy = eval_policy.RandomSplitEvaluationPolicy

# Tests ----------------------------------------------------------------------

def test_invalid_evaluation_ratio_raises():
    with pytest.raises(ValueError):
        RandomSplitEvaluationPolicy(evaluation_ratio=0.0)
    with pytest.raises(ValueError):
        RandomSplitEvaluationPolicy(evaluation_ratio=1.0)
    with pytest.raises(ValueError):
        RandomSplitEvaluationPolicy(evaluation_ratio=-0.1)

def test_initialize_split_requires_at_least_two():
    loader = DataLoader([1])
    state = GEPAState(i=0)
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=0)
    with pytest.raises(ValueError):
        policy.get_selection_batch(loader, state)

def test_split_deterministic_and_changes_with_iteration():
    ids = [0, 1, 2, 3]
    loader = DataLoader(ids)
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=123)

    state = GEPAState(i=0)
    eval_batch_1 = policy.get_eval_batch(loader, state)
    sel_batch_1 = policy.get_selection_batch(loader, state)

    # repeated calls for same iteration should yield identical batches
    eval_batch_1b = policy.get_eval_batch(loader, state)
    sel_batch_1b = policy.get_selection_batch(loader, state)
    assert eval_batch_1 == eval_batch_1b
    assert sel_batch_1 == sel_batch_1b

    # ensure disjoint and union equals full set
    assert set(eval_batch_1).isdisjoint(set(sel_batch_1))
    assert set(eval_batch_1).union(set(sel_batch_1)) == set(ids)

    # expected counts: n_eval = max(1, int(len(ids) * ratio))
    expected_n_eval = max(1, int(len(ids) * 0.5))
    assert len(eval_batch_1) == expected_n_eval
    assert len(sel_batch_1) == len(ids) - expected_n_eval

    # different iteration should produce a (likely) different split when seed is used
    state.i = 1
    eval_batch_2 = policy.get_eval_batch(loader, state)
    sel_batch_2 = policy.get_selection_batch(loader, state)

    # It is possible (rare) that the shuffle yields the same partition; assert at least one differs.
    assert (eval_batch_2 != eval_batch_1) or (sel_batch_2 != sel_batch_1)

def test_get_best_program_and_valset_score():
    # Construct three programs over same two tasks 'a' and 'b'
    prog_scores = [
        {"a": 0.4, "b": 0.4},  # prog 0
        {"a": 1.0, "b": 0.0},  # prog 1 (expected best)
        {"a": 0.1, "b": 0.1},  # prog 2
    ]

    avg_map = {
        0: (0.4, 2),
        1: (0.5, 2),
        2: (0.1, 2),
    }

    state = GEPAState(i=0, prog_candidate_val_subscores=prog_scores, avg_map=avg_map)
    policy = RandomSplitEvaluationPolicy(evaluation_ratio=0.5, seed=42)

    best_idx = policy.get_best_program(state)
    assert best_idx == 1

    # valset score proxies to state's get_program_average_val_subset()[0]
    assert policy.get_valset_score(1, state) == 0.5