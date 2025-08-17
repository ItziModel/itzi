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

from typing import Tuple, Mapping

import numpy as np


class DomainData:
    """Store raster domain information. Alike GRASS region."""

    def __init__(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        rows: int,
        cols: int,
        crs_wkt: str,
    ):
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.rows = rows
        self.cols = cols
        self.crs_wkt = crs_wkt

        if self.north < self.south:
            raise ValueError(f"north must be superior to south. {self.north=}, {self.south=}")
        if self.east < self.west:
            raise ValueError(f"east must be superior to west. {self.east=}, {self.west=}")

        self.nsres = (self.north - self.south) / self.rows
        self.ewres = (self.east - self.west) / self.cols
        self.cell_area = self.ewres * self.nsres
        self.cell_shape = (self.ewres, self.nsres)
        self.shape = (self.rows, self.cols)
        self.cells = self.rows * self.cols

    def is_in_domain(self, *, x: float, y: float) -> bool:
        """For a given coordinate pair(x, y),
        return True is inside the domain, False otherwise.
        """
        bool_x = self.west < x < self.east
        bool_y = self.south < y < self.north
        return bool(bool_x and bool_y)

    def coordinates_to_pixel(self, *, x: float, y: float) -> Tuple[float, float] | None:
        """For a given coordinate pair(x, y),
        return True is inside the domain, False otherwise.
        """
        if not self.is_in_domain(x=x, y=y):
            return None
        else:
            norm_row = (y - self.south) / (self.north - self.south)
            row = int(np.round((1 - norm_row) * (self.rows - 1)))

            norm_col = (x - self.west) / (self.east - self.west)
            col = int(np.round(norm_col * (self.cols - 1)))

            return (row, col)

    def get_coordinates(self) -> Mapping[str, np.ndarray]:
        """Return x and y coordinates as numpy arrays representing the center of each cell.
        Returns:
            A mapping with 'x' and 'y' keys containing 1D numpy arrays.
            - 'x': array of shape (cols,) with x-coordinates of cell centers
            - 'y': array of shape (rows,) with y-coordinates of cell centers
        """
        # X coordinates: from west + half cell width, incrementing by cell width
        x_coords = np.linspace(self.west + self.ewres / 2, self.east - self.ewres / 2, self.cols)

        # Y coordinates: from north - half cell height, decrementing by cell height
        # (raster coordinates typically go from north to south)
        y_coords = np.linspace(self.north - self.nsres / 2, self.south + self.nsres / 2, self.rows)

        return {"x": x_coords, "y": y_coords}

    def __repr__(self):
        return (
            f"north={self.north}, "
            f"south={self.south}. "
            f"east={self.east}, "
            f"west={self.west}, "
            f"rows={self.rows}, "
            f"cols={self.cols}, "
            f"nsres={self.nsres}, "
            f"ewres={self.ewres}, "
            f"cell_area={self.cell_area}, "
            f"cell_shape={self.cell_shape}, "
            f"shape={self.shape}, "
            f"cells={self.cells}, "
        )
