# AGENTS.md

## Common commands
- Run a single test: `uv run pytest tests/my_test.py`
- The GRASS tests need `--forked`: `uv run pytest --forked tests/`
- Enforce code formatting: `uvx ruff format .`

## Code style
- Use python type hints
- Use pydantic BaseModel instead of dataclass

## General comments
- The project uses `uv`. To run a command in the correct environment, use `uv run`
- Running the whole test suite is slow. Do it only after all the specific tests are passing, as a final check.
