
'''
author: Beiming Tang
date: 10/28/2026
DAFCOM - Washington DC to Boston subset
'''

import numpy as np
import xarray as xr
from datetime import datetime as dt, timedelta
from pathlib import Path

# ============================================================
# 0. Initialization
# ============================================================

dir_ = Path('/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/')
dir_2 = Path('/scratch3/NCEPDEV/stmp/Beiming.Tang/Data/')

dir_year = '2026'
dir_month = '2026_05'

date = 1
start_hour = 0
number_of_hours = 96

year = int(dir_year)
month = int(dir_month.split('_')[1])

hour_start = (date - 1) * 24 + start_hour
hour_end = hour_start + number_of_hours

# Washington DC - Boston
#LAT_MIN, LAT_MAX = 37.1, 44.1
#LON_MIN, LON_MAX = -77.6, -70.6

# Washington DC only (approximately 27 × 27 grids)
#LAT_MIN, LAT_MAX = 38.76, 39.02
#LON_MIN, LON_MAX = -77.15, -76.89

# CONUS domain (west half)
LAT_MIN, LAT_MAX = 25, 49
LON_MIN, LON_MAX = -125, -95

# CONUS domain (east half)
#LAT_MIN, LAT_MAX = 25, 49
#LON_MIN, LON_MAX = -95, -65

# ============================================================
# 1. Determine spatial indices (read coordinates only)
# ============================================================

reference_file = (
    dir_ / '0_prepare_Meteo/step2_regrid'
    / dir_year / dir_month / 'evaporation_0p01.nc'
)

with xr.open_dataset(reference_file) as ds:
    lat_all = ds['lat'].to_numpy()
    lon_all = ds['lon'].to_numpy()

# Support longitude in either -180..180 or 0..360
lon_check = (lon_all + 180) % 360 - 180

if lat_all.ndim == 1 and lon_all.ndim == 1:

    iy = np.flatnonzero(
        (lat_all >= LAT_MIN) & (lat_all <= LAT_MAX)
    )
    ix = np.flatnonzero(
        (lon_check >= LON_MIN) & (lon_check < LON_MAX)
    )
    original_shape = (len(lat_all), len(lon_all))

elif lat_all.ndim == 2 and lon_all.ndim == 2:

    mask = (
        (lat_all >= LAT_MIN) &
        (lat_all <= LAT_MAX) &
        (lon_check >= LON_MIN) &
        (lon_check <= LON_MAX)
    )
    iy, ix = np.where(mask)
    original_shape = lat_all.shape

else:
    raise ValueError("Unsupported coordinate dimensions")

if len(iy) == 0 or len(ix) == 0:
    raise ValueError("Requested DC-Boston region is outside grid")

ys = slice(int(iy.min()), int(iy.max()) + 1)
xs = slice(int(ix.min()), int(ix.max()) + 1)

ny = ys.stop - ys.start
nx = xs.stop - xs.start

shape_3d = (number_of_hours, ny, nx)

if lat_all.ndim == 1:
    LAT = lat_all[ys]
    LON = lon_all[xs]
else:
    LAT = lat_all[ys, xs]
    LON = lon_all[ys, xs]

print("Original shape:", original_shape)
print("Subset shape:", (ny, nx))
print("96-hour shape:", shape_3d)

# ============================================================
# 2. General loading functions
# ============================================================

def crop_spatial(da):
    """Crop the last two dimensions before loading data."""

    if da.shape[-2:] != original_shape:
        raise ValueError(
            f"Grid mismatch for {da.name}: "
            f"{da.shape[-2:]} != {original_shape}"
        )

    return da.isel({
        da.dims[-2]: ys,
        da.dims[-1]: xs
    })


def load_dynamic(relative_path, var_name):
    """Load 96 hours, DC-Boston region only."""

    filename = dir_ / relative_path

    with xr.open_dataset(filename) as ds:
        da = ds[var_name]

        if da.ndim != 3:
            raise ValueError(
                f"{var_name} expected 3D, got {da.shape}"
            )

        da = da.isel({
            da.dims[0]: slice(hour_start, hour_end)
        })

        da = crop_spatial(da)

        if da.shape[0] != number_of_hours:
            raise ValueError(
                f"{var_name}: requested 96 hours, "
                f"got {da.shape[0]}"
            )

        return da.to_numpy().astype(
            np.float32, copy=False
        )


def load_static(relative_path, var_name):
    """Read one 2D map, broadcast to 96 hours."""

    filename = dir_2 / relative_path

    with xr.open_dataset(filename) as ds:
        da = ds[var_name]

        # Static data may contain singleton/time dimensions
        for dim in da.dims[:-2]:
            da = da.isel({dim: 0})

        da = crop_spatial(da)

        arr = da.to_numpy().astype(
            np.float32, copy=False
        )

    # No physical 96-fold copy
    return np.broadcast_to(arr[None, :, :], shape_3d)


# ============================================================
# 3. Meteorology
# ============================================================

print("1 Meteorology")

meteo_dir = (
    f"0_prepare_Meteo/step2_regrid/"
    f"{dir_year}/{dir_month}/"
)

BLH = load_dynamic(
    meteo_dir + 'pblh_0p01.nc', 'pblh'
)

D2M = load_dynamic(
    meteo_dir + 'specific_humidity_2m_0p01.nc',
    'specific_humidity_2m'
)

E = load_dynamic(
    meteo_dir + 'evaporation_0p01.nc', 'evaporation'
)

SP = load_dynamic(
    meteo_dir + 'surface_pressure_0p01.nc',
    'surface_pressure'
)

T2M = load_dynamic(
    meteo_dir + 't2m_0p01.nc', 't2m'
)

TP = load_dynamic(
    meteo_dir + 'precipitation_0p01.nc',
    'precipitation'
)

U10 = load_dynamic(
    meteo_dir + 'u10_0p01.nc', 'u10'
)

V10 = load_dynamic(
    meteo_dir + 'v10_0p01.nc', 'v10'
)

print("BLH shape:", BLH.shape)


# ============================================================
# 4. AOD
# ============================================================

print("2 AOD")

aod_file = (
    f"1_prerare_AOD/step2_regrid/{dir_year}/"
    f"aod_{month:02d}_0p01.nc"
)

AOD = load_dynamic(aod_file, 'aod')

print("AOD shape:", AOD.shape)


# ============================================================
# 5. Land Use Cover
# ============================================================

print("3 Land Use Cover")

Land_Use_Cover = load_static(
    'LandUseCover_MODIS/2023/LUC_0p01_regrid_data.nc',
    'luc'
)

print("LUC shape:", Land_Use_Cover.shape)


# ============================================================
# 6. Elevation
# ============================================================

print("4 Elevation")

Elevation = load_static(
    'Elevation_SRTM/elevation_final_regrid_0p01.nc',
    'Elevation'
)

print("Elevation shape:", Elevation.shape)


# ============================================================
# 7. Population
# ============================================================

print("5 Population")

Population = load_static(
    'Population_LandScan/2023/Population_0p01_regrid_data.nc',
    'Population'
)

print("Population shape:", Population.shape)


# ============================================================
# 8. Chemistry
# ============================================================

print("6 Chemistry")

chem_dir = (
    f"5_prepare_Pollutant/step2_regrid/"
    f"{dir_year}/{dir_month}/"
)

UFS_AQM_PM25 = load_dynamic(
    chem_dir + 'pm25_0p01.nc',
    'pm25'
)

print("PM25 shape:", UFS_AQM_PM25.shape)


# ============================================================
# 9. Anthropogenic Emissions (NEMO 2019)
# ============================================================

print("7 NEMO Emissions")

emis_dir = f"Emis_NEMO/2019/2019_{month:02d}/"

E_BC_hourly = load_static(
    emis_dir + 'E_bc_0p01_regrid_data.nc', 'E_bc'
)

E_NH3_hourly = load_static(
    emis_dir + 'E_nh3_0p01_regrid_data.nc', 'E_nh3'
)

E_NOX_hourly = load_static(
    emis_dir + 'E_nox_0p01_regrid_data.nc', 'E_nox'
)

E_VOC_hourly = load_static(
    emis_dir + 'E_voc_0p01_regrid_data.nc', 'E_voc'
)

E_PM25_hourly = load_static(
    emis_dir + 'E_pm25_0p01_regrid_data.nc', 'E_pm25'
)

E_SO2_hourly = load_static(
    emis_dir + 'E_so2_0p01_regrid_data.nc', 'E_so2'
)

print("E_SO2 shape:", E_SO2_hourly.shape)


# ============================================================
# 10. Latitude, Longitude, Distance
# ============================================================

print("8 Distance")

distance_file = dir_ / '7_prepare_Distance/Distance_0p01.nc'


def load_distance(da):
    """Handle static 2D or time-dependent 3D distance data."""

    if da.ndim == 3:
        nt = da.shape[0]

        if nt >= hour_end:
            da = da.isel({
                da.dims[0]: slice(hour_start, hour_end)
            })
        elif nt == number_of_hours:
            pass
        elif nt == 1:
            da = da.isel({da.dims[0]: 0})
        else:
            raise ValueError(
                f"Unexpected time dimension for {da.name}: {nt}"
            )

    elif da.ndim != 2:
        raise ValueError(f"Unexpected shape: {da.shape}")

    da = crop_spatial(da)
    arr = da.to_numpy().astype(np.float32, copy=False)

    if arr.ndim == 2:
        arr = np.broadcast_to(arr[None, :, :], shape_3d)

    return arr


with xr.open_dataset(distance_file) as ds:
    LAT_hourly = load_distance(ds['LAT'])
    LON_hourly = load_distance(ds['LON'])

    D1 = load_distance(ds['D1'])
    D2 = load_distance(ds['D2'])
    D3 = load_distance(ds['D3'])
    D4 = load_distance(ds['D4'])
    D5 = load_distance(ds['D5'])

print("D1 shape:", D1.shape)


# ============================================================
# 11. Time Inputs
# ============================================================

print("9 Time")

start_date = dt(year, month, date, start_hour)

date_list = [
    start_date + timedelta(hours=i)
    for i in range(number_of_hours)
]

hour_list = np.array(
    [t.hour for t in date_list],
    dtype=np.uint8
)

day_list = np.array(
    [t.timetuple().tm_yday for t in date_list],
    dtype=np.uint16
)

hour_matrix = np.broadcast_to(
    hour_list[:, None, None],
    shape_3d
)

day_matrix = np.broadcast_to(
    day_list[:, None, None],
    shape_3d
)

print("Hour shape:", hour_matrix.shape)
print("Day shape:", day_matrix.shape)


# ============================================================
# 12. Final Verification
# ============================================================

print("\n========== FINAL SHAPES ==========")

variables = {
    'BLH': BLH,
    'D2M': D2M,
    'E': E,
    'SP': SP,
    'T2M': T2M,
    'TP': TP,
    'U10': U10,
    'V10': V10,
    'AOD': AOD,
    'Land_Use_Cover': Land_Use_Cover,
    'Elevation': Elevation,
    'Population': Population,
    'UFS_AQM_PM25': UFS_AQM_PM25,
    'E_BC_hourly': E_BC_hourly,
    'E_NH3_hourly': E_NH3_hourly,
    'E_NOX_hourly': E_NOX_hourly,
    'E_VOC_hourly': E_VOC_hourly,
    'E_PM25_hourly': E_PM25_hourly,
    'E_SO2_hourly': E_SO2_hourly,
    'LAT_hourly': LAT_hourly,
    'LON_hourly': LON_hourly,
    'D1': D1,
    'D2': D2,
    'D3': D3,
    'D4': D4,
    'D5': D5,
    'hour_matrix': hour_matrix,
    'day_matrix': day_matrix
}

for name, arr in variables.items():
    print(f"{name:20s}: {arr.shape}")

#print("\nDC-Boston subset preparation complete!")
print("\nDC subset preparation complete!")
