# Robust LLM-as-a-Judge Evaluation

Run teacher response evaluation 3 times to get a more robust win-lose-tie result. 

## Overview

다수결로 (즉 2번 이상) 선택되어야 win

세 개의 결과가 모두 불일치하면 tie

그렇지 않으면 lose

## Files

- `evaluate_teacher_responses.py` - Single-run evaluation script
- `run_robust_evaluation.py` - New robust evaluation script that runs multiple evaluations
- `evaluation_prompt.txt` - Judge prompt template (required)

## Usage

### Basic Usage

```bash
python run_robust_evaluation.py --input_file endpoint_responses_20250620_182458.json --output_dir evaluation_results
```

### With Custom Parameters

```bash
python run_robust_evaluation.py \
    --input_file endpoint_responses_20250620_182458.json \
    --output_dir evaluation_results \
    --max_evaluations 50
```

### Using Existing Evaluation Files

If you already have evaluation files from previous runs, you can aggregate them directly:

```bash
python run_robust_evaluation.py \
    --existing_files eval1.json eval2.json eval3.json \
    --output_dir evaluation_results
```

This is useful when:
- You want to combine results from evaluations run at different times
- You have evaluation files from different sources
- You want to re-aggregate with different logic without re-running evaluations

### Parameters

- `--input_file`: Input JSON file containing response comparisons (required for new evaluations)
- `--output_dir`: Output directory for evaluation results (default: `data_collection_logs/evaluations`)
- `--max_evaluations`: Maximum number of evaluations per run (optional, for testing)
- `--existing_files`: Exactly 3 existing evaluation JSON files to aggregate (alternative to running new evaluations)

## Aggregation Logic

The system uses the following rules to determine final verdicts:

1. **Majority Rule**: If 2 or more runs agree on a verdict, that becomes the final verdict
2. **Tie Rule**: If all 3 runs have different verdicts, the final verdict is "tie"
3. **Confidence Levels**:
   - `unanimous`: All runs agree
   - `majority`: 2+ runs agree
   - `no_consensus`: All runs differ (tie case)

## Output Format

The robust evaluation produces a JSON file with:

```json
{
  "summary": {
    "evaluation_completed": "2025-01-01T12:00:00.000000",
    "num_evaluation_runs": 3,
    "total_subsets": 200,
    "successful_evaluations": 195,
    "final_verdict_distribution": {
      "chat": 118,
      "simple-chat": 73,
      "tie": 4
    },
    "confidence_distribution": {
      "unanimous": 150,
      "majority": 45,
      "no_consensus": 5
    },
    "ours_win_rate": 0.605,
    "baseline_win_rate": 0.374,
    "tie_rate": 0.021,
    "unanimous_agreement_rate": 0.75
  },
  "aggregated_results": {
    "subset_id_1": {
      "verdicts": ["chat", "chat", "simple-chat"],
      "verdict_counts": {"chat": 2, "simple-chat": 1},
      "final_verdict": "chat",
      "confidence": "majority"
    }
  }
}
```

## Testing

Run the test script to verify aggregation logic:

```bash
python test_robust_evaluation.py
```

## Requirements

- Python 3.7+
- All dependencies from `evaluate_teacher_responses.py`
- `evaluation_prompt.txt` file must exist in the same directory

## Notes

- Each evaluation run is independent and uses the same input file
- The script handles failures gracefully - if one run fails, it continues with the remaining runs
- Results from individual runs are preserved for analysis
- The system is designed to be more robust against LLM variability and prompt sensitivity