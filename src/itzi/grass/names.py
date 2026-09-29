"""
Copyright (C) 2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


def derived_record_name(dataset_id: str, record_index: int) -> str:
    """Return the shared raster/vector child ID for one output record."""
    if record_index < 0:
        raise ValueError("record_index must be non-negative")
    name, separator, mapset = dataset_id.partition("@")
    return f"{name}_{record_index:04d}{separator}{mapset}"


def derived_drainage_table_names(vector_id: str) -> tuple[str, str]:
    """Return the table names created for a drainage vector child."""
    name = vector_id.partition("@")[0]
    return (f"{name}_node", f"{name}_link")
