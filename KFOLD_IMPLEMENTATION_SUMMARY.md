# K-Fold Rotation Evaluation Policy - Implementation Summary

## Overview
Successfully implemented K-fold cross-validation evaluation policy for GEPA that dramatically reduces overfitting by rotating through different validation folds across iterations.

## Changes Made

### 1. Core Implementation (`src/gepa/strategies/eval_policy.py`)
- Added `KFoldRotationEvaluationPolicy` class
- Partitions validation set into K folds (default: 5)
- Rotates evaluation folds each iteration to prevent overfitting
- Each iteration evaluates 1/K of the validation data (one fold)

### 2. API Integration (`src/gepa/api.py`)
- Added "kfold" as a string option for `val_evaluation_policy` parameter
- Updated type hints: `Literal["full_eval", "kfold"]`
- Updated error messages and documentation
- Default K-fold uses 5 folds when "kfold" string is passed

### 3. Comprehensive Testing
- `tests/test_kfold_eval_policy.py`: 8 unit tests for the policy class
- `tests/test_kfold_api_option.py`: 3 integration tests for API usage
- All 12 tests pass ✅

## Usage Examples

### Simple String Option (Recommended)
```python
result = gepa.optimize(
    seed_candidate={"system_prompt": "..."},
    trainset=train_data,
    valset=val_data,
    adapter=my_adapter,
    val_evaluation_policy="kfold",  # 5 folds by default
    max_metric_calls=100,
)
```

### Custom Number of Folds
```python
from gepa.strategies.eval_policy import KFoldRotationEvaluationPolicy

policy = KFoldRotationEvaluationPolicy(num_folds=3)
result = gepa.optimize(
    val_evaluation_policy=policy,
    # ... other parameters
)
```

## How It Works

**Partitioning**: V = V₁ ∪ V₂ ∪ ... ∪ Vₖ

**Rotation Schedule**:
- Iteration 0 (seed): Evaluate on V₁, select candidates using V₂...Vₖ
- Iteration 1: Evaluate on V₂, select candidates using V₁,V₃...Vₖ
- Iteration K-1: Evaluate on Vₖ, select candidates using V₁...Vₖ₋₁
- Iteration K: Back to V₁ (cycle repeats)

**Key Insight**: At each iteration, one fold is reserved for evaluation of the newly proposed program, while the remaining (K-1) folds are used for candidate selection (determining which program to evolve from).

**Final Reporting**: At the end of optimization, the best program and validation scores are calculated using ALL available validation data (not just candidate selection folds) to provide an accurate assessment of final performance.

## Benefits

1. **Reduced Overfitting**: Candidate selection and evaluation use disjoint data during optimization, preventing overfitting to a single split
2. **Better Generalization**: Each iteration evaluates on different data and selects candidates based on different data
3. **Accurate Final Reporting**: Final results use complete validation set for true performance assessment
4. **Efficient**: Evaluates only 1/K of validation data per iteration (1/5 with default 5 folds), reducing computational cost
5. **Drop-in Replacement**: Works as a simple string option or custom instance

## Files Modified/Created

Modified:
- `src/gepa/strategies/eval_policy.py` (+91 lines)
- `src/gepa/api.py` (+4 lines, updated docs)

Created:
- `tests/test_kfold_eval_policy.py` (157 lines, 8 tests)
- `tests/test_kfold_api_option.py` (105 lines, 3 tests)
- `KFOLD_USAGE_EXAMPLE.py` (usage demonstration)

## Test Results
```
12 tests passed in 0.08s
- 8 unit tests for KFoldRotationEvaluationPolicy
- 3 integration tests for API string option
- 1 existing incremental eval policy test
```

## Comparison with Other Policies

| Policy | Eval Coverage per Iteration | Selection Coverage | Overfitting Risk | Computation |
|--------|----------------------------|-------------------|------------------|-------------|
| `full_eval` | 100% | 100% | High | High |
| `kfold` (K=5) | 20% (1 fold) | 80% (4 folds) | Low | Low |
| Custom sampling | Variable | Variable | Medium | Low-Medium |

## When to Use K-Fold

✅ **Use "kfold" when:**
- Running many GEPA iterations (>10)
- Have moderate to large validation set (>20 examples)
- Overfitting is a concern
- Want balance between coverage and generalization

✅ **Use "full_eval" when:**
- Small validation set (<20 examples)
- Running few iterations (<5)
- Need complete validation coverage every time

## Documentation

The implementation includes:
- Comprehensive docstrings
- Type hints throughout
- Parameter validation (num_folds ≥ 2)
- Clear error messages
- Usage examples
