# Task 291: NBA Luxury Tax Calculator - IFE Fix

## Problem Summary
The task asks to calculate NBA luxury tax but the hidden implementation uses a simplified, arbitrary formula:
- Base tax: 15% of excess salary (not actual NBA brackets)
- Repeat offender: `(excess_salary // 5_000_000) * 0.05` surcharge

The actual NBA CBA uses tiered progressive brackets, but the benchmark expects this simplified formula that is not specified in the task description.

## IFE Type
**Hidden Information Design Issue / Specification Mismatch**

The hidden implementation contains arbitrary constants (0.15 base rate, 5M step, 0.05 increment) that:
1. Are not stated in the task description
2. Do not match real NBA CBA rules
3. Cannot be reliably communicated by the simulated user

## Fix Approach
Add clarification to specify the expected simplified formula. This makes the task **fair** (specifies the expected algorithm) without making it **easy** (agent still needs to implement the logic correctly).

## Evidence
- Hidden code: `tax_due = excess_salary * 0.15` and `tax_due += (excess_salary // 5000000) * 0.05`
- Task description only mentions "luxury tax due to the NBA's collective bargaining agreement" without specifying the formula
- Test cases: `calculate_luxury_tax(40000000, 70000000, 80000000, False) == 1500000` (10M excess * 0.15 = 1.5M)
