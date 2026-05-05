#### Load in the correct opacity table for the 2M_\odot ZAMS star for the X = 0.7, Y = 0.28, Z = 0.02 composition. The table is interpolated and for a given inputted temperature and density, the opacity is reported. Corresponds to solution of Homework 3 and Steps 3 and 4 of the instructions. 

import numpy as np
from pathlib import Path
from astropy.io import ascii
from scipy.interpolate import RegularGridInterpolator
import numpy.lib.recfunctions

opacity_table = "/Users/kmanj/Downloads/opacity_x0.7y0.28z0.02.txt"

if not Path(opacity_table).exists():
    opacity_table = Path(__file__).resolve().parent / "opacity_x0.7y0.28z0.02.txt"

def _load_opacity_table(filename):
    # Load in the opacity table and populate the opacity grid. Apply a regular grid interpolator to use within the solver. 
    # Inputs: filename of opacity table
    # Outputs: interpolator, logT grid, logR grid
    
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find the line where the table header begins (identified by "logT")
    header_start = None
    for i, line in enumerate(lines):
        if "logT" in line:
            header_start = i
            break

    data_start = header_start + 1
    table = ascii.read(filename, header_start=header_start, data_start=data_start)

    # Extract logR grid from column names (skip first column = logT)
    logR_grid = np.array([float(col) for col in table.colnames[1:]], dtype=float)
    # Extract logT grid from first column
    logT_grid = np.array(table["logT"], dtype=float)

    opacity_grid = np.lib.recfunctions.structured_to_unstructured(table.as_array())
    opacity_grid = np.delete(opacity_grid, 0, axis=1)
    opacity_grid = np.array(opacity_grid, dtype=float)

    # Mask extreme/unphysical opacity values (outside valid OPAL range)
    opacity_grid[opacity_grid > 8] = np.nan

    # Apply the interpolator to clean regions between opacity values within the table
    interp = RegularGridInterpolator((logT_grid, logR_grid),opacity_grid,bounds_error=False,fill_value=np.nan)
    return interp, logT_grid, logR_grid

# Compute the interpolation table with the logT and logR grids once to avoid computational cost of persistent interpolating
_opacity_interp, _logT_grid, _logR_grid = _load_opacity_table(opacity_table)

def get_opacity(T, rho):
    # Compute the opacity value from the interpolated grid for a given temperature and density
    # Inputs: temperature, density
    # Outputs: opacity value (kappa)
    
    T   = np.maximum(10**3.75, T)    
    T6  = T * 1e-6
    R   = np.maximum(1e-8, rho / T6**3)

    logT_target = np.log10(T)
    logR_target = np.log10(R)

    # Find the closest interpolated opacity for the density and temperature provided
    if (logT_target < _logT_grid.min() or logT_target > _logT_grid.max() or
            logR_target < _logR_grid.min() or logR_target > _logR_grid.max()):
        return np.nan

    # Report the opactiy value 
    log_kappa = _opacity_interp([[logT_target, logR_target]])[0]
    return 10**log_kappa
