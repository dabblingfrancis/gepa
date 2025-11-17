"""
Example: Using K-Fold Rotation Evaluation Policy with GEPA

This example demonstrates how to use the "kfold" option for val_evaluation_policy
to reduce overfitting during GEPA optimization.
"""

import gepa

# Prepare your data
trainset = [
    {"id": i, "input": f"train_{i}", "expected": f"output_{i}"}
    for i in range(20)
]

valset = [
    {"id": i, "input": f"val_{i}", "expected": f"output_{i}"}
    for i in range(15)
]

# Option 1: Use the simple string option "kfold" (uses default 5 folds)
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    adapter=my_adapter,  # Your adapter
    reflection_lm=my_llm,  # Your LLM
    val_evaluation_policy="kfold",  # ✨ Use K-fold rotation with 5 folds
    max_metric_calls=100,
)

# Option 2: Use a custom KFoldRotationEvaluationPolicy instance for more control
from gepa.strategies.eval_policy import KFoldRotationEvaluationPolicy

custom_policy = KFoldRotationEvaluationPolicy(num_folds=3)  # Use 3 folds instead of 5

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    adapter=my_adapter,
    reflection_lm=my_llm,
    val_evaluation_policy=custom_policy,  # ✨ Use custom K-fold policy
    max_metric_calls=100,
)

# Option 3: Use the default full evaluation (evaluates all validation data every iteration)
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    adapter=my_adapter,
    reflection_lm=my_llm,
    val_evaluation_policy="full_eval",  # Or None, or omit this parameter
    max_metric_calls=100,
)

print("Optimization complete!")
print(f"Best candidate: {result.best_candidate}")
print(f"Best validation score: {result.best_val_score}")
