"""Test that the 'kfold' string option works in gepa.optimize()."""

import gepa
from gepa.core.adapter import EvaluationBatch


class DummyAdapter:
    """Simple test adapter."""

    def __init__(self):
        self.propose_new_texts = self._propose_new_texts

    def evaluate(self, batch, candidate, capture_traces=False):
        weight = int(candidate.get("system_prompt", "weight=0").split("=")[-1])
        outputs = [{"id": item["id"], "weight": weight} for item in batch]
        scores = [min(1.0, (weight + 1) / max(item.get("difficulty", 1), 1)) for item in batch]
        trajectories = [{"score": score} for score in scores] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [{"score": score} for score in eval_batch.scores]
        return dict.fromkeys(components_to_update, records)

    def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        weight = int(candidate.get("system_prompt", "weight=0").split("=")[-1])
        return dict.fromkeys(components_to_update, f"weight={weight + 1}")


def test_kfold_string_option_in_optimize(tmp_path):
    """Test that val_evaluation_policy='kfold' works in gepa.optimize()."""
    trainset = [
        {"id": i, "difficulty": 2, "split": "train"}
        for i in range(5)
    ]
    valset = [
        {"id": i, "difficulty": 3, "split": "val"}
        for i in range(15)
    ]

    adapter = DummyAdapter()

    # Test with "kfold" string option
    result = gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=None,
        candidate_selection_strategy="current_best",
        max_metric_calls=20,
        run_dir=str(tmp_path / "run_kfold"),
        val_evaluation_policy="kfold",  # Use string option
    )

    # Verify that optimization ran
    assert len(result.val_subscores) > 0
    assert result.best_candidate is not None

    # Verify that different validation examples were used across iterations
    # With K-fold rotation, each iteration should see different subsets
    all_evaluated_ids = set()
    for scores in result.val_subscores:
        all_evaluated_ids.update(scores.keys())

    # Should have evaluated some validation examples
    assert len(all_evaluated_ids) > 0


def test_full_eval_string_option_in_optimize(tmp_path):
    """Test that val_evaluation_policy='full_eval' works in gepa.optimize()."""
    trainset = [
        {"id": i, "difficulty": 2, "split": "train"}
        for i in range(5)
    ]
    valset = [
        {"id": i, "difficulty": 3, "split": "val"}
        for i in range(10)
    ]

    adapter = DummyAdapter()

    # Test with "full_eval" string option
    result = gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=None,
        candidate_selection_strategy="current_best",
        max_metric_calls=15,
        run_dir=str(tmp_path / "run_full"),
        val_evaluation_policy="full_eval",  # Use string option
    )

    # Verify that optimization ran
    assert len(result.val_subscores) > 0
    assert result.best_candidate is not None


def test_invalid_eval_policy_string_raises_error(tmp_path):
    """Test that an invalid string for val_evaluation_policy raises an error."""
    trainset = [{"id": i, "difficulty": 2, "split": "train"} for i in range(3)]
    valset = [{"id": i, "difficulty": 3, "split": "val"} for i in range(5)]

    adapter = DummyAdapter()

    try:
        gepa.optimize(
            seed_candidate={"system_prompt": "weight=0"},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=None,
            max_metric_calls=5,
            run_dir=str(tmp_path / "run_invalid"),
            val_evaluation_policy="invalid_option",  # Invalid string
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "val_evaluation_policy should be one of" in str(e)
