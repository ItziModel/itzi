# coding=utf8
"""
Copyright (C) 2016-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from datetime import datetime
import csv
import numbers
import dataclasses

from itzi.data_containers import MassBalanceData


class MassBalanceLogger:
    """Writes pre-calculated mass balance data to a CSV file."""

    def __init__(
        self,
        file_name: str,
        start_time: datetime,
        temporal_type: str,
    ):
        """Initializes the logger and creates the output file with headers."""
        if temporal_type not in ["absolute", "relative"]:
            raise ValueError(f"unknown temporal type <{temporal_type}>")
        self.temporal_type = temporal_type
        self.start_time = start_time
        self.fields = [f.name for f in dataclasses.fields(MassBalanceData)]
        self.file_name = self._set_file_name(file_name)
        self._create_file()

    def _set_file_name(self, file_name: str) -> str:
        """Generate output file name"""
        if not file_name:
            file_name = "{}_stats.csv".format(str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))
        return file_name

    def _create_file(self) -> None:
        """Create a csv file and write headers"""
        with open(self.file_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log(self, report_data: MassBalanceData) -> None:
        """Writes a single line of data to the CSV file."""
        line_to_write = {}

        for key, value in dataclasses.asdict(report_data).items():
            if value != value:  # test for NaN
                line_to_write[key] = "-"
            elif "percent_error" == key:
                line_to_write[key] = f"{value:.2%}"
            elif "simulation_time" == key and "relative" == self.temporal_type:
                line_to_write[key] = report_data.simulation_time - self.start_time
            elif isinstance(value, numbers.Real) and not isinstance(value, int):
                line_to_write[key] = f"{value:.3f}"
            else:
                line_to_write[key] = value

        with open(self.file_name, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(line_to_write)
