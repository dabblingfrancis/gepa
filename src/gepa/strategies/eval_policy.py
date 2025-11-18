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


class SubsampleEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that samples a subset of validation examples for evaluation.
    
    Prioritizes validation examples that have been evaluated least frequently,
    enabling efficient candidate selection with reduced evaluation costs.
    """

    def __init__(self, batch_size: int = 5):
        """Initialize the subsample evaluation policy.
        
        Args:
            batch_size: Number of validation examples to sample for each evaluation.
                       Must be a positive integer.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Return ids sorted by how often they've been evaluated.
        
        Prefers validation examples that have been evaluated least frequently,
        particularly prioritizing examples not yet evaluated.
        
        Args:
            loader: Data loader containing validation examples.
            state: Current GEPA optimization state.
            target_program_idx: Optional program index (unused in this implementation).
            
        Returns:
            List of validation IDs to evaluate, up to batch_size length.
        """
        all_ids = list(loader.all_ids())
        if not all_ids:
            return []

        # Create an ordering index to maintain stable ordering for ties
        order_index = {val_id: idx for idx, val_id in enumerate(all_ids)}
        valset_evaluations = state.valset_evaluations

        def sort_key(val_id: DataId):
            eval_count = len(valset_evaluations.get(val_id, []))
            return (eval_count, order_index[val_id])

        # Sort by evaluation count (ascending), then by original order
        ordered_ids = sorted(all_ids, key=sort_key)
        
        # Return up to batch_size examples, or all if fewer are available
        batch = ordered_ids[: self.batch_size] or ordered_ids

        return batch

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
    "SubsampleEvaluationPolicy",
]
