# Benchmark Notes

## Prompt Update + Stricter Test-Heavy Data Filtering

Latest observed result after:

- switching prompt rendering away from ChatML-style wrappers
- using benchmark-like prompt formatting
- filtering out more test-heavy completions from training data

HumanEval results:

| benchmark | pass@1 |
| --- | ---: |
| humaneval (base tests) | 0.591 |
| humaneval+ (base + extra tests) | 0.543 |

Takeaway:

- best non-baseline result so far
- still below the base model
- prompt changes helped less than data cleaning, but neither has fully closed the gap to baseline
