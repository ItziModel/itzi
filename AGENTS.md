# Itzi flood model

## Common commands
- Run a single test: `uv run pytest tests/my_test.py`
- The GRASS tests need `--forked`: `uv run pytest --forked tests/grass`. This is not needed for core tests.
- Enforce code formatting: `uvx ruff format .`

## Code style
- Use python type hints. When a function that does not yet use hints is substantially edited, take the opportunity to add type hints.
- Since the arguments types and return types are already documented by the hints, there's no need to duplicate this information in the docstrings.
- Apart from particular cases, use pydantic BaseModel instead of dataclass
- Place imports at the top of the file. Only break this rule to prevent heavy imports in a rarely used function (for example, CLI options).

## General comments
- The project uses `uv`. To run a command in the correct environment, use `uv run`
- Running the whole test suite is slow. Do it only after all the specific tests are passing, as a final check.
