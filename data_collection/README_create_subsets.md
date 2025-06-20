# Conversation Subset Generator

The `create_subsets.py` script generates random subsets of conversation data from teacher-student conversations, saved in "session_*.json".  Each subset ends with a user utterance. 

## Usage

### Basic usage
```bash
python create_subsets.py
```
Generates 100 random subsets from `data_collection_logs/` and saves them to `data_collection_logs/subsets/`

### Custom options
```bash
# Generate specific number of subsets
python create_subsets.py --num_subsets 50

# Use custom input/output directories
python create_subsets.py --input_dir data_collection_logs --output_dir my_subsets

# Set random seed for reproducible results
python create_subsets.py --num_subsets 200 --seed 42
```

### Output

The script creates:
- Individual subset files: `subset_{id}.json`
- Summary file: `subsets_summary_{timestamp}.json` with generation statistics
- Preserved original metadata (student personas, traits, project IDs)

### Requirements

- Input: JSON files starting with "session" in the specified directory
- 형식: user-assistant를 반복하는 대화 패턴, user로 시작해야 함