# Task 47: Non-profit Organization Frontend

## IFE Analysis

**Task Type:** Frontend Design (CLIP similarity evaluation)

**IFE Identified:** Yes - Simulated User Visual Comparison Loop Broken

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "The frontend-design task instructions require the agent to provide raw HTML wrapped in `<html>`... However, the execution environment/parser rejects raw HTML unless it appears inside a triple-backtick code fence that matches a Python-oriented regex"

2. **Evaluation 4 (Grade 1.0):** "The simulated user channel is supposed to compare the rendered agent snippet against the ground-truth image. However, in multiple turns it claims it cannot see the agent image"

### Root Cause

The ColBench frontend evaluation pipeline has two interacting defects:

1. **Parser/Environment Mismatch:** The smolagents-based execution environment expects code in Python fenced blocks (`````python```) with tool calls, but the task specification says to output "raw html code wrapped in `<html>` tag". This creates ambiguity about the correct output format.

2. **Broken Visual Feedback Loop:** The `HTML_USER_PROMPT` says the simulated user "will be given an image made by the agent and a ground-truth image", but the image rendering/passing mechanism fails, causing the simulated user to respond "The agent did not provide any visualization" or "I can't see the image provided by the agent."

### Impact

Without functional visual feedback, agents cannot iteratively converge to the exact ground-truth design. The CLIP similarity evaluation then penalizes designs that don't match pixel-perfect expectations that were never communicated.

## Fix Strategy

This is an **infrastructure-level IFE** that cannot be fully fixed via task-level overrides. However, we can document the issue and provide clarifications.

### No Code Fix Required

This IFE is in the evaluation infrastructure (visual comparison loop), not the task specification. The task itself (design a non-profit website) is reasonable.

**Recommended infrastructure fix (out of scope for item-level fixing):**
- Ensure the simulated user receives the rendered agent HTML as an image before comparison
- Fix the environment to properly handle HTML snippets in the iterative design loop

### Why No Fix File Created

Creating an `instruction_override.json` or `evaluation_override.json` would not address the core issue:
- The problem is the visual feedback mechanism, not the task description
- Lowering CLIP similarity thresholds would be "making it easy" not "making it fair"
- The task difficulty (collaborative design) should remain intact
