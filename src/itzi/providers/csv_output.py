"""Application-owned CSV output with explicit first-creation ownership."""

from __future__ import annotations

import csv
import numbers
from pathlib import Path

from itzi_core.data_containers import MassBalanceData
from itzi_core.providers.base import MassBalanceOutputProvider


class ExclusiveCSVMassBalanceOutputProvider(MassBalanceOutputProvider):
    """Write statistics without silently truncating a pre-existing destination."""

    def __init__(
        self, file_name: str | Path, *, overwrite: bool, float_format: str = ".3f"
    ) -> None:
        self.fields = list(MassBalanceData.model_fields.keys())
        self.file_name = Path(file_name)
        self.file_name.parent.mkdir(parents=True, exist_ok=True)
        self.float_format = float_format
        mode = "w" if overwrite else "x"
        with self.file_name.open(mode, newline="") as file_obj:
            csv.DictWriter(file_obj, fieldnames=self.fields).writeheader()

    def log(self, report_data: MassBalanceData) -> None:
        line_to_write = {}
        for key, value in report_data.model_dump().items():
            if value != value:  # noqa: PLR0124
                line_to_write[key] = "-"
            elif isinstance(value, numbers.Real) and not isinstance(value, int):
                line_to_write[key] = format(value, self.float_format)
            else:
                line_to_write[key] = value
        with self.file_name.open("a", newline="") as file_obj:
            csv.DictWriter(file_obj, fieldnames=self.fields).writerow(line_to_write)
