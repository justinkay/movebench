# ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.

# General coding guidelines

Keep code simple. Keep lines of code to a minimum for any new implementation. Do not add features that were not explicitly asked for. Do not add code that was not explicitly asked for. Do not reimplement functionality that can be found elsewhere or in commonly-used dependencies. Follow DRY principles and avoid writing overly engineered spaghetti code. 

Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.

Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.

Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.

Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.

Don't make things unnecessarily reusable. Keep YAGNI and KISS in mind.

# Stylistic preferences

Keep imports at the top of the file. Do not lazy load imports. Do not include a bunch of try/excepts for imports -- assume the user has imported the required libraries. Use isort and the black code formatter as style inspirations.

# Environment

Use the movebench micromamba environment located in /data/vision/beery/scratch/justin/micromamba/envs.

# Testing discipline

- For multi-phase work, create/update tests and fixtures before implementing each milestone. Prefer having at least one test that fails on the pre-change behavior and passes after the fix.
- Run tests after each milestone before moving on. At minimum run `python -m unittest -v`; also run any smaller, milestone-specific test command(s) when available.
- Do not claim a task is completed (or mark an ExecPlan `Progress` checkbox complete) until the relevant tests have been run and the outputs reviewed.
- If a test cannot be run or is skipped due to environment constraints (credentials, network, missing large datasets), record the exact reason and do not treat it as verified in the current environment.
