# Repository Workflow

- Maintain a repo-root `PLAN.md` for every non-trivial task.
- Update `PLAN.md` before substantial edits, keep it current while working, and mark completed items when done.
- Do not remove `PLAN.md`; treat it as a lightweight running execution log for the current change set.
- Make sure to use env `conda activate hal` for any python executions.
- Be resilient. If any agent that returned empty or failed sessions, just restart them, possibly that's just a transient issue.
