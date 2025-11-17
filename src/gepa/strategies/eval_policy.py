"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState, ProgramIdx


@runtime_checkable
class EvaluationPolicy(Protocol[DataId, DataInst]):  # type: ignore
    """Strategy for choosing validation ids to evaluate and identifying best programs for validation instances."""

    @abstractmethod
    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Select examples for evaluation for a program"""
        ...

    @abstractmethod
    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return "best" program given all validation results so far across candidates"""
        ...

    @abstractmethod
    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        ...


class FullEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that evaluates all validation instances every time."""

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]


class KFoldRotationEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that partitions validation set into K folds and rotates through them each iteration.

    At each iteration t, one fold V_i is used for candidate selection, and the remaining
    folds V \\ V_i are used for evaluation. After each iteration, the fold rotates to prevent
    overfitting to any single validation split.

    Iteration 1 → use fold V1 for selection, V2...VK for evaluation
    Iteration 2 → use fold V2 for selection, V1,V3...VK for evaluation
    ...
    Iteration K → use fold VK for selection, V1...VK-1 for evaluation
    Iteration K+1 → V1 again (cycle repeats)
    """

    def __init__(self, num_folds: int = 5):
        """Initialize K-fold rotation policy.

        Args:
            num_folds: Number of folds to partition the validation set into (K).
        """
        if num_folds < 2:
            raise ValueError("num_folds must be at least 2")
        self.num_folds = num_folds
        self._folds: list[list[DataId]] = []
        self._all_ids: list[DataId] = []
        self._initialized = False

    def _initialize_folds(self, loader: DataLoader[DataId, DataInst]) -> None:
        """Partition the validation set into K folds."""
        self._all_ids = list(loader.all_ids())
        n = len(self._all_ids)

        # Create K roughly equal-sized folds
        self._folds = []
        fold_size = n // self.num_folds
        remainder = n % self.num_folds

        start = 0
        for i in range(self.num_folds):
            # Distribute remainder across first folds
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            self._folds.append(self._all_ids[start:end])
            start = end

        self._initialized = True

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Return evaluation fold: all ids except the current candidate-selection fold.

        The fold used for evaluation rotates with each iteration. At iteration t,
        fold (t % K) is used for candidate selection, and the remaining folds are
        used for evaluation.
        """
        if not self._initialized:
            self._initialize_folds(loader)

        if not self._all_ids:
            return []

        # Determine current iteration from the number of programs
        iteration = len(state.prog_candidate_val_subscores)
        current_fold_idx = iteration % self.num_folds

        # Evaluation fold = V \ V_i (all folds except the candidate-selection fold)
        eval_ids = []
        for i, fold in enumerate(self._folds):
            if i != current_fold_idx:
                eval_ids.extend(fold)

        return eval_ids

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
    "KFoldRotationEvaluationPolicy",
]
