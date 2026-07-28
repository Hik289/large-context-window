# Locomo Test Cases Documentation

This document explains how to use the configurable test case system for the Locomo memory evaluation framework.

## Overview

The test case system allows you to:
- Define multiple test scenarios in a YAML configuration file
- Run individual test cases or batch process multiple tests
- Easily add new test cases without modifying code
- Compare results across different test scenarios

## File Structure

```
unittest/
├── test_cases.yaml          # Test case configurations
├── test_case_1.py          # Main test runner (updated)
├── run_test_cases.py       # Command-line utility
└── README.md               # This documentation
```

## Configuration File (test_cases.yaml)

The `test_cases.yaml` file contains all test case definitions:

```yaml
test_cases:
  - name: "Caroline Research Test"
    description: "Test case for Caroline's research activities"
    conversation_idx: 0
    session_idx: 2
    num_sessions: 1
    question: "What did Caroline research?"
```

### Test Case Parameters

- **name**: Unique identifier for the test case
- **description**: Human-readable description of what the test evaluates
- **conversation_idx**: Index of the conversation to use from the dataset
- **session_idx**: Index of the session within the conversation
- **num_sessions**: Number of sessions to process
- **question**: The question to ask the memory system

## Usage Methods

### Method 1: Using test_case_1.py (Updated)

The original script now loads test cases from YAML:

```python
# Run default test case
python test_case_1.py

# The script automatically loads from test_cases.yaml
# and runs the default test case specified in the config
```

### Method 2: Using run_test_cases.py (Command Line)

This utility provides more control over test execution:

```bash
# List all available test cases
python run_test_cases.py --list

# Run a specific test case
python run_test_cases.py --test "Caroline Research Test"

# Run multiple test cases in batch
python run_test_cases.py --batch

# Run specific test cases in batch
python run_test_cases.py --batch --tests "Caroline Research Test" "Speaker Interaction Test"
```

## Adding New Test Cases

To add a new test case, edit `test_cases.yaml`:

```yaml
test_cases:
  # Existing test cases...
  
  - name: "Your New Test Case"
    description: "Description of what this test evaluates"
    conversation_idx: 1
    session_idx: 0
    num_sessions: 1
    question: "Your test question?"
```

## Batch Testing Configuration

Configure batch testing behavior in `test_cases.yaml`:

```yaml
batch_config:
  run_all_tests: false
  selected_tests: 
    - "Caroline Research Test"
    - "Speaker Interaction Test"
  output_results: true
  compare_results: true
```

## Example Output

When running a test case, you'll see output like:

```
============================================================
Running Test Case: Caroline Research Test
Description: Test case for Caroline's research activities
Question: What did Caroline research?
============================================================

==== Building AgentMemory Memory ====
[Memory building progress...]

==== Test Results ====
Question: What did Caroline research?
Answer: Caroline researched machine learning algorithms for natural language processing.
Response: Based on the conversation history...
Speaker 1 (Caroline) Memories: 15 items
Speaker 2 (David) Memories: 12 items
```

## Integration with Existing Workflow

The new system is backward compatible:
- Existing scripts continue to work
- Default test case maintains original behavior
- New functionality is opt-in

## Benefits

1. **Maintainability**: Test cases are defined in configuration, not code
2. **Scalability**: Easy to add new test scenarios
3. **Flexibility**: Run single tests or batch process multiple scenarios
4. **Comparison**: Easily compare results across different test cases
5. **Documentation**: Each test case includes description and context

## Advanced Usage

### Custom Test Case Files

You can use different YAML files:

```python
test_config = load_test_cases("custom_test_cases.yaml")
```

### Programmatic Test Execution

```python
from test_case_1 import load_test_cases, run_single_test_case

# Load configuration
test_config = load_test_cases()

# Run specific test
test_case = get_test_case_by_name(test_config, "Caroline Research Test")
result = run_single_test_case(cfg, test_case)

# Access results
print(f"Answer: {result['answer']}")
print(f"Memories: {len(result['speaker_1_memories'])}")
```

This system provides a robust foundation for systematic memory evaluation across multiple test scenarios.