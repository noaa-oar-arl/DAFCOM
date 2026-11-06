"""Regular half-open grids and nonoverlapping tile ownership."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .config import Grid


@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def name(self):
        return f"y{self.y0}-{self.y1}_x{self.x0}-{self.x1}"

    def slices(self, grid: Grid, halo=0):
        ny, nx = grid.shape
        return (
            slice(max(0, self.y0 - halo), min(ny, self.y1 + halo)),
            slice(max(0, self.x0 - halo), min(nx, self.x1 + halo)),
        )


def tiles(grid):
    ny, nx = grid.shape
    ty, tx = grid.tile_shape
    return [
        Tile(y, min(y + ty, ny), x, min(x + tx, nx))
        for y in range(0, ny, ty)
        for x in range(0, nx, tx)
    ]


def coordinates(grid):
    ny, nx = grid.shape
    return (
        grid.north - np.arange(ny) * grid.resolution,
        grid.west + np.arange(nx) * grid.resolution,
    )


def target_grid(grid, tile, halo=1):
    lat, lon = coordinates(grid)
    ys, xs = tile.slices(grid, halo)
    return xr.Dataset(
        coords={
            "lat": ("lat", lat[ys], {"units": "degrees_north"}),
            "lon": ("lon", lon[xs], {"units": "degrees_east"}),
        }
    )


def owners(grid, lat, lon):
    latitude, longitude = coordinates(grid)
    valid = (
        (lat >= latitude[-1])
        & (lat <= latitude[0])
        & (lon >= longitude[0])
        & (lon <= longitude[-1])
    )
    y = np.floor((grid.north - lat) / grid.resolution + 1e-8).astype(int)
    x = np.floor((lon - grid.west) / grid.resolution + 1e-8).astype(int)
    return valid, y, x


def distances(lat, lon, reference):
    west, south, east, north = reference
    anchors = [
        (west, north),
        (east, north),
        (west, south),
        (east, south),
        ((west + east) / 2, (south + north) / 2),
    ]
    latitude, longitude = np.deg2rad(lat), np.deg2rad(lon)
    result = []
    for x, y in anchors:
        a = (
            np.sin((np.deg2rad(y) - latitude) / 2) ** 2
            + np.cos(latitude)
            * np.cos(np.deg2rad(y))
            * np.sin((np.deg2rad(x) - longitude) / 2) ** 2
        )
        result.append(6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
    return result
