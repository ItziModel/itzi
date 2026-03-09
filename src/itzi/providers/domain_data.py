"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import numpy as np

from pydantic import BaseModel, ConfigDict, computed_field, model_validator


class DomainData(BaseModel):
    """Store raster domain information. Alike GRASS region."""

    model_config = ConfigDict(frozen=True)

    north: float
    south: float
    east: float
    west: float
    rows: int
    cols: int
    crs_wkt: str

    @model_validator(mode="after")
    def check_bounds(self) -> "DomainData":
        if self.north < self.south:
            raise ValueError(f"north must be superior to south. {self.north=}, {self.south=}")
        if self.east < self.west:
            raise ValueError(f"east must be superior to west. {self.east=}, {self.west=}")
        return self

    # --- Computed fields (derived, read-only properties) ---

    @computed_field
    @property
    def nsres(self) -> float:
        return (self.north - self.south) / self.rows

    @computed_field
    @property
    def ewres(self) -> float:
        return (self.east - self.west) / self.cols

    @computed_field
    @property
    def cell_area(self) -> float:
        return self.ewres * self.nsres

    @computed_field
    @property
    def cell_shape(self) -> tuple[float, float]:
        return (self.ewres, self.nsres)

    @computed_field
    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @computed_field
    @property
    def cells(self) -> int:
        return self.rows * self.cols

    def is_in_domain(self, *, x: float, y: float) -> bool:
        """Return True if (x, y) is inside the domain, False otherwise."""
        return self.west < x < self.east and self.south < y < self.north

    def coordinates_to_pixel(self, *, x: float, y: float) -> tuple[int, int] | None:
        """Return (row, col) pixel indices for a given (x, y) coordinate, or None if outside domain."""
        if not self.is_in_domain(x=x, y=y):
            return None
        norm_row = (y - self.south) / (self.north - self.south)
        row = int(np.round((1 - norm_row) * (self.rows - 1)))
        norm_col = (x - self.west) / (self.east - self.west)
        col = int(np.round(norm_col * (self.cols - 1)))
        return (row, col)

    def get_coordinates(self) -> dict[str, np.ndarray]:
        """Return x and y coordinates as 1D arrays of cell centers."""
        x_coords = np.linspace(
            self.west + self.ewres / 2, self.east - self.ewres / 2, num=self.cols
        )
        y_coords = np.linspace(
            self.north - self.nsres / 2, self.south + self.nsres / 2, num=self.rows
        )
        return {"x": x_coords, "y": y_coords}

    def __repr__(self):
        return repr(self.model_dump())
