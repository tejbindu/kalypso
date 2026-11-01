# Implementation instructions

These instructions apply to all agents working in this repository, including Luna implementation agents.

## Code and change quality

- Develop in small, focused increments that are easy to understand and review. Each approved increment should form a coherent commit.
- Keep code clean and follow the repository's conventions and established best practices.
- Add functions and helpers alongside their first real caller. Avoid commits that introduce unused runtime functions ahead of the feature path that will use them; keep implementation history in the order a reader can follow it.
- Add comments wherever they help explain intent, assumptions, invariants, or non-obvious behavior. Comments may be as detailed as necessary.
- Run focused checks appropriate to each increment and report their results and any limitations.
- Keep implementation plans and design discussions in `plans/`; consult the relevant approved design before implementing.

## Mandatory review before committing

- The user wants to review every change. After completing one reviewable increment, stop and report the changed files, behavior, validation results, and any open questions.
- Do not continue to the next increment while the current increment awaits review.
- Make requested revisions, then stop again for review.
- Do not stage or commit implementation changes until the user explicitly approves committing that increment. Approval of a plan or authorization to implement is not approval to commit.
- When a commit is approved, write a detailed commit message explaining the problem, the resulting behavior, relevant design decisions, and validation.
- Preserve unrelated user changes and never include them in a commit without authorization.

## Agent workflow

- The user prefers Luna for block-filter implementation and code-related discussion to reduce usage. Keep routine implementation work and follow-up discussion with the Luna agent; consult the primary agent when requested or when a design issue needs escalation.
- Start with one small implementation increment, not the entire block-filter feature in one change. Follow `plans/BLOCK_FILTER_DESIGN.md` and its implementation sequence.
- Agents must honor the same review pauses regardless of who delegates the work. A handoff is not permission to commit or to proceed past a review boundary.
