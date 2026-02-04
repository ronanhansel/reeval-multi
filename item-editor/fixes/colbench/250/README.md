# Task 250: Forum Thread Deletion Function

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Simulated User Hallucinates Requirements Contradicting Hidden Code

### Problem Description
```
Can you write a Python function that determines which old threads in a forum to delete,
considering both the age of the thread and its importance, to strike a balance between
preserving knowledge and improving database query speed?
The signature: def delete_old_threads(threads, importance_threshold, num_to_keep)
```

### Hidden Ground Truth
```python
def delete_old_threads(threads, importance_threshold, num_to_keep):
    threads_to_delete = [thread for thread in threads
                         if thread['importance'] < importance_threshold
                         and len(threads) - threads.index(thread) > num_to_keep]
    return threads_to_delete
```

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "The simulated user is instructed to only answer using hidden information, but the hidden information is a single-line reference implementation... the simulated user hallucinated additional requirements/fields (e.g., `created_at` ISO string, 'keep N most recent' and sort by date), which are not present in hidden info"

2. **Evaluation 5 (Grade 1.0):** "The dialogue channel cannot reliably convey the true target behavior because the hidden info is code and the simulator is not constrained to it... simulated user replies contradict hidden info: 'Each thread is a dictionary with keys like `id`, `created_at`, `importance`' and 'The function should return a list of thread IDs to delete'"

### Root Cause - Simulated User Hallucination

**What hidden code actually does:**
- Uses list position (`threads.index(thread)`) as implicit age
- No `created_at` field - age is determined by position in list
- Returns full thread dictionaries, not IDs
- Weird condition: `len(threads) - threads.index(thread) > num_to_keep`

**What simulated user told agent:**
- "Each thread has fields: 'id', 'created_at', 'importance'" (hallucinated `created_at`)
- "age is measured as now minus created_at" (contradicts hidden code's positional age)
- "function should return a list of thread IDs to delete" (contradicts returning dicts)

**The user is making up requirements that don't exist in hidden code!**

This is a critical benchmark flaw: the simulated user is supposed to only answer from hidden information, but instead invents plausible but incorrect specifications.

### Why This is NOT Fixable at Task Level

**Cannot be fixed at task level** because:
1. The issue is in the simulated user's behavior, not the task spec
2. We cannot control what the simulated user hallucinates
3. Even fixing the problem description won't prevent user contradictions

The hidden code's use of `threads.index(thread)` for age is also problematic:
- Using list index as age proxy is fragile (fails with duplicates)
- Natural interpretation would expect a timestamp field

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error (simulated user hallucination). The benchmark would need to:
1. Fix the simulated user to accurately extract facts from code
2. Or add the actual schema (no `created_at`, positional age) to task description
3. Or restructure hidden code to use actual timestamp field

Documenting as unfixable IFE (simulated user contradicts hidden ground truth).
