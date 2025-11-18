"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import random

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


class RandomSplitEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that randomly splits validation set into selection and evaluation subsets.
    
    The validation set is split into two subsets:
    - Selection subset: Reserved for candidate selection (not used during optimization)
    - Evaluation subset: Used during optimization for evaluating candidates
    
    The best program is determined by centering scores per task and selecting
    the program with the highest average centered score.
    """

    def __init__(self, selection_ratio: float = 0.5, seed: int | None = None):
        """Initialize the random split evaluation policy.
        
        Args:
            selection_ratio: Fraction of validation set to use for selection (default 0.5).
                           Must be between 0 and 1.
            seed: Random seed for reproducible splits. If None, uses system random.
        """
        if not 0 < selection_ratio < 1:
            raise ValueError("selection_ratio must be between 0 and 1")
        self.selection_ratio = selection_ratio
        self.rng = random.Random(seed)
        self._selection_ids: list[DataId] | None = None
        self._evaluation_ids: list[DataId] | None = None

    def _initialize_split(self, loader: DataLoader[DataId, DataInst]) -> None:
        """Initialize the random split if not already done."""
        if self._selection_ids is not None:
            return
        
        all_ids = list(loader.all_ids())
        if not all_ids:
            self._selection_ids = []
            self._evaluation_ids = []
            return
        
        # Shuffle and split
        shuffled_ids = all_ids.copy()
        self.rng.shuffle(shuffled_ids)
        
        split_point = int(len(shuffled_ids) * self.selection_ratio)
        self._selection_ids = shuffled_ids[:split_point]
        self._evaluation_ids = shuffled_ids[split_point:]

    def _compute_task_means(self, state: GEPAState) -> dict[DataId, float]:
        """Compute mean score per task across all programs.
        
        Args:
            state: Current GEPA optimization state.
            
        Returns:
            Dictionary mapping task IDs to their mean scores across all programs.
        """
        # Collect all task IDs that have been evaluated
        all_task_ids: set[DataId] = set()
        for scores in state.prog_candidate_val_subscores:
            all_task_ids.update(scores.keys())
        
        # Calculate mean score per task across all programs
        task_means: dict[DataId, float] = {}
        for task_id in all_task_ids:
            task_scores = []
            for scores in state.prog_candidate_val_subscores:
                if task_id in scores:
                    task_scores.append(scores[task_id])
            if task_scores:
                task_means[task_id] = sum(task_scores) / len(task_scores)
        
        return task_means

    def _compute_centered_scores(
        self, scores: dict[DataId, float], task_means: dict[DataId, float]
    ) -> list[float]:
        """Compute centered scores for a program.
        
        Args:
            scores: Dictionary of task scores for a program.
            task_means: Dictionary mapping task IDs to their mean scores.
            
        Returns:
            List of centered scores (score - mean for each task).
        """
        centered_scores = []
        for task_id, score in scores.items():
            if task_id in task_means:
                centered_score = score - task_means[task_id]
                centered_scores.append(centered_score)
        return centered_scores

    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Return the evaluation subset for evaluation during optimization.
        
        Args:
            loader: Data loader containing validation examples.
            state: Current GEPA optimization state.
            target_program_idx: Optional program index (unused in this implementation).
            
        Returns:
            List of validation IDs in the evaluation subset.
        """
        self._initialize_split(loader)
        return self._evaluation_ids or []

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program with the highest average centered score.
        
        Scores are centered per task by subtracting the mean score across all
        programs for each task. This accounts for varying task difficulties.
        
        Args:
            state: Current GEPA optimization state.
            
        Returns:
            Index of the best program.
        """
        if not state.prog_candidate_val_subscores:
            return -1
        
        # Compute task means
        task_means = self._compute_task_means(state)
        if not task_means:
            return -1
        
        # Calculate centered scores for each program
        best_idx = -1
        best_centered_score = float("-inf")
        best_coverage = -1
        
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            if not scores:
                continue
            
            # Calculate centered score for this program
            centered_scores = self._compute_centered_scores(scores, task_means)
            
            if not centered_scores:
                continue
            
            avg_centered_score = sum(centered_scores) / len(centered_scores)
            coverage = len(centered_scores)
            
            # Select program with highest average centered score
            # Use coverage as tiebreaker
            if avg_centered_score > best_centered_score or (
                avg_centered_score == best_centered_score and coverage > best_coverage
            ):
                best_centered_score = avg_centered_score
                best_idx = program_idx
                best_coverage = coverage
        
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the average centered score of the program on the valset.
        
        Args:
            program_idx: Index of the program to score.
            state: Current GEPA optimization state.
            
        Returns:
            Average centered score for the program.
        """
        if program_idx < 0 or program_idx >= len(state.prog_candidate_val_subscores):
            return float("-inf")
        
        scores = state.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf")
        
        # Compute task means
        task_means = self._compute_task_means(state)
        
        # Calculate centered scores for this program
        centered_scores = self._compute_centered_scores(scores, task_means)
        
        if not centered_scores:
            return float("-inf")
        
        return sum(centered_scores) / len(centered_scores)


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
    "RandomSplitEvaluationPolicy",
]
