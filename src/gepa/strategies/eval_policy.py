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

    @abstractmethod
    def get_selection_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState
    ) -> list[DataId]:
        """Select validation IDs to use for candidate selection.
        
        Returns validation IDs that should be used for selecting candidates.
        Policies like RandomSplitEvaluationPolicy return only the selection subset,
        while FullEvaluationPolicy returns all validation IDs.
        """
        raise NotImplementedError


class FullEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that evaluates all validation instances every time."""

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())
    
    def get_selection_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Return all validation IDs for candidate selection.
        
        For FullEvaluationPolicy, the selection batch is the same as the full validation set.
        
        Args:
            loader: Data loader containing validation examples.
            state: Current GEPA optimization state.
            target_program_idx: Optional program index (unused in this implementation).
            
        Returns:
            List of all validation IDs.
        """
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

    @abc.abstractmethod
    def get_selection_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState
    ) -> list[DataId]:
        """Return validation IDs to use for candidate selection."""
        pass


class RandomSplitEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that randomly splits validation set into selection and evaluation subsets.

    With each iteration, the valset is randomly split into two subsets:
    - Selection subset: Used for candidate selection
    - Evaluation subset: Used to evaluate the mutated prompt (if the prompt revision was successful on trainset)

    The split prevents lucky prompts from dominating, as they need to generalize to a different part of the valset. 

    The best program is determined by computing advantages (GRPO-style) per task on the valset
    and selecting the program with the highest average advantage. Using advantages instead of raw scores removes the possibility that a program was lucky to be evaluated on an easier subset of the valset. 
    """

    def __init__(self, evaluation_ratio: float = 0.5, seed: int | None = None):
        """Initialize the random split evaluation policy.
        
        Args:
            evaluation_ratio: Fraction of validation set to use for evaluation (default 0.5).
                           Must be between 0 and 1.
            seed: Random seed for reproducible splits. If None, uses system random.
        """
        if not 0 < evaluation_ratio < 1:
            raise ValueError("evaluation_ratio must be between 0 and 1")
        self.evaluation_ratio = evaluation_ratio
        self.rng = random.Random(seed)
        self._selection_ids: list[DataId] | None = None
        self._evaluation_ids: list[DataId] | None = None

    def _initialize_split(self, loader: DataLoader[DataId, DataInst]) -> None:
        """Initialize the random split if not already done."""
        if self._selection_ids is not None:
            return
        
        all_ids = list(loader.all_ids())
        
        # Require at least 2 validation items for RandomSplitEvaluationPolicy
        if len(all_ids) < 2:
            raise ValueError(
                f"RandomSplitEvaluationPolicy requires at least 2 validation items, got {len(all_ids)}"
            )
        
        # Shuffle and split
        shuffled_ids = all_ids.copy()
        self.rng.shuffle(shuffled_ids)
        
        split_point = int(len(shuffled_ids) * self.evaluation_ratio)
        
        # Ensure at least 1 evaluation item and 1 selection item
        # Ensure at least 1 item in both subsets
        if split_point == 0:
            split_point = 1
        
        self._evaluation_ids = shuffled_ids[:split_point]
        self._selection_ids = shuffled_ids[split_point:]

    def _compute_task_means(self, state: GEPAState) -> dict[DataId, float]:
        """Compute mean score per task across all programs on the validation set.
        
        Args:
            state: Current GEPA optimization state.
            
        Returns:
            Dictionary mapping task IDs to their mean scores across all programs.
        """
        # Calculate mean score per task across all programs (for all validation tasks)
        task_means: dict[DataId, float] = {}
        
        # Get all unique task IDs from all programs
        all_task_ids = set()
        for scores in state.prog_candidate_val_subscores:
            all_task_ids.update(scores.keys())
        
        # Compute mean for each task
        for task_id in all_task_ids:
            task_scores = []
            for scores in state.prog_candidate_val_subscores:
                if task_id in scores:
                    task_scores.append(scores[task_id])
            if task_scores:
                task_means[task_id] = sum(task_scores) / len(task_scores)
        
        return task_means

    def _compute_advantages(
        self, scores: dict[DataId, float], task_means: dict[DataId, float]
    ) -> list[float]:
        """Compute advantages for a program.
        
        Args:
            scores: Dictionary of task scores for a program.
            task_means: Dictionary mapping task IDs to their mean scores.
            
        Returns:
            List of advantages (score - mean for each task).
        """
        advantages = []
        for task_id, score in scores.items():
            if task_id in task_means:
                advantage = score - task_means[task_id]
                advantages.append(advantage)
        return advantages

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
        """Pick the program with the highest average advantage on the valset.
        
        Advantages are computed per task by subtracting the mean score across all 
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
        
        # Calculate advantages for each program
        best_idx = -1
        best_advantage = float("-inf")
        best_coverage = -1
        
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            if not scores:
                continue
            
            # Calculate advantage for this program
            advantages = self._compute_advantages(scores, task_means)
            
            if not advantages:
                continue
            
            avg_advantage = sum(advantages) / len(advantages)
            coverage = len(advantages)
            
            # Select program with highest average advantage
            # Use coverage as tiebreaker
            if avg_advantage > best_advantage or (
                avg_advantage == best_advantage and coverage > best_coverage
            ):
                best_advantage = avg_advantage
                best_idx = program_idx
                best_coverage = coverage
        
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]

    def get_advantage(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the average advantage of the program on the selection subset.
        
        Args:
            program_idx: Index of the program to score.
            state: Current GEPA optimization state.
            
        Returns:
            Average advantage for the program on the selection subset.
        """
        if program_idx < 0 or program_idx >= len(state.prog_candidate_val_subscores):
            return float("-inf")
        
        scores = state.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf")
        
        # Compute task means (for selection subset only)
        task_means = self._compute_task_means(state)
        
        # Calculate advantages for this program (only for selection subset tasks)
        advantages = self._compute_advantages(scores, task_means)
        
        if not advantages:
            return float("-inf")
        
        return sum(advantages) / len(advantages)
    
    def get_selection_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Return the selection subset for candidate selection.
        
        Args:
            loader: Data loader containing validation examples.
            state: Current GEPA optimization state.
            target_program_idx: Optional program index (unused in this implementation).
            
        Returns:
            List of validation IDs in the selection subset.
        """
        self._initialize_split(loader)
        return self._selection_ids or []


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
    "RandomSplitEvaluationPolicy",
]
