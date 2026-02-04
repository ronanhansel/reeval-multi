# Task 79 Fix Analysis

## Task Type
This task appears in both backend and frontend benchmarks:
- **Backend Task 79**: Battle game determining winner based on health/attack/defense
- **Frontend Task 79**: Food Company website design

## Issues Identified

### Backend Task 79: Simulated User Information Extraction Failure

**Issue**: The simulated user is instructed to only answer using hidden information and say "I don't know" otherwise. However, when the agent asks about the exact damage mechanics (which ARE clearly defined in the hidden reference code), the simulated user responds "I don't know the specific details for the damage formula or turn order."

**Evidence from rubric**:
- Agent asked: "Is damage = attacker_attack - defender_defense (floored at 0)? ... How is defender damage calculated ... Turn order..."
- Simulated user replied: "I don't know the specific details for the damage formula or turn order."
- Hidden info clearly defines: `defender_health -= max(attacker_attack - defender_defense, 0)` and `attacker_health -= max(defender_defense - attacker_attack, 0)`

**Root Cause**: The simulated user prompt tells the user to say "I don't know" if the answer can't be found in hidden info, but the hidden info is CODE. The simulated user (GPT-4o) should be able to read the code and extract the mechanics, but it's failing to translate code structure into explicit factual answers.

**Verdict**: This is borderline - in one evaluation the run succeeded (grade=0), in another it failed (grade=1). The task CAN succeed if the agent proposes defaults that happen to match, or if the simulated user successfully extracts the mechanics in some runs. However, the inconsistency in the simulated user's ability to extract information from code IS a formation issue.

### Frontend Task 79: Parser/Format Mismatch (CLEAR IFE)

**Issue**: The evaluation harness enforces a Python fenced code block regex `(?:py|python)?\s*\n(.*?)\n` even for this frontend HTML/Tailwind task.

**Evidence**:
- Task requirement: "Write the code inside a tag `<html>`" and "The answer should be a piece of raw html code wrapped in `<html>` tag"
- Parser error: "Your code snippet is invalid, because the regex pattern ` ```(?:py|python)?\s*\n(.*?)\n``` ` was not found in it."
- Agent provided valid HTML starting with `<html>` but it was rejected

**Root Cause**: The harness is configured to parse code using a Python-oriented regex, which contradicts the frontend task's HTML output requirement.

## Fix Decision

**NO FIX CREATED** for the following reasons:

1. **Backend Task 79**: The issues are inconsistent - some runs succeed, some fail. This appears to be variance in the simulated user's ability to interpret code, not a systematic benchmark defect. Agents can work around this by proposing mechanics and getting confirmation.

2. **Frontend Task 79**: The parser mismatch is a HARNESS-LEVEL issue affecting all frontend tasks, not something that can be fixed with instruction/evaluation overrides at the task level. This requires a fix to the ColBench evaluation infrastructure itself, not a per-task fix.

The parser issue should be reported as a benchmark infrastructure bug. The fix would need to be in:
- `hal-harness/hal/benchmarks/colbench.py` or the agent harness
- NOT in task-level overrides

## Recommendation

1. File a bug report for the ColBench benchmark regarding the Python-regex parsing requirement for HTML tasks
2. No task-level fixes are appropriate here - the issues are either:
   - Variance in simulated user behavior (not fixable at task level)
   - Infrastructure-level parsing issues (needs harness fix, not task fix)
