# Itzi flood model

## Common commands
- Run a single test: `uv run pytest tests/my_test.py`
- Due to a bug in GRASS, tests will fails if the mapset is changed between tests. Run each test file independently to prevent this. Tests that need to run in a separate process are marked with `@pytest.mark.forked`.
- Enforce code formatting: `uvx ruff format .`

## Code style
- Use python type hints. When a function that does not yet use hints is substantially edited, take the opportunity to add type hints.
- Do not quote class names in hints. Use `from __future__ import annotations` when necessary.
- Since the arguments types and return types are already documented by the hints, there's no need to duplicate this information in the docstrings.
- Apart from particular cases, use pydantic BaseModel instead of dataclass
- Place imports at the top of the file. Only break this rule to prevent heavy imports in a rarely used function (for example, CLI options).

## General comments
- The project uses `uv`. To run a command in the correct environment, use `uv run`
- Running the whole test suite is slow. Do it only after all the specific tests are passing, as a final check.
