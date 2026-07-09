#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:28:00 2025

@author: corden
"""

import xarray as xr
import pandas as pd
import numpy as np

def regularize_time_grid(obj, freq='10s', tol=None, numeric_agg='mean'): 
    """
    Regularize the time coordinate of a Dataset or DataArray to a fixed grid.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Input data with a 'time' coordinate.
    freq : str, default '5S'
        Frequency for the regular time grid (pandas-style, e.g. '5S', '1min').
    tol : str or pd.Timedelta or None
        Maximum tolerance for snapping times to the nearest grid point.
        Default: half the freq.
    numeric_agg : {'mean','first','last'}
        Aggregation for multiple samples falling into the same time bin.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Object with regular 5s-spaced time coordinate, missing data filled with NaN.
    """

    is_dataarray = isinstance(obj, xr.DataArray)
    ds = obj.to_dataset(name='__temp__') if is_dataarray else obj.copy()

    if 'time' not in ds.coords:
        raise ValueError("Input must have a 'time' coordinate.")

    # ensure datetime64
    orig_times = pd.to_datetime(ds['time'].values)
    rounded = orig_times.round(freq)

    # tolerance default = half the freq
    if tol is None:
        tol = pd.to_timedelta(freq) / 2
    else:
        tol = pd.to_timedelta(tol)

    # snap within tolerance
    diffs = np.abs(rounded - orig_times)
    keep_mask = diffs <= tol
    rounded_masked = np.array([t if m else pd.NaT for t, m in zip(rounded, keep_mask)],
                              dtype='datetime64[ns]')

    ds = ds.assign_coords(time_rounded=('time', rounded_masked))

    # full regular index
    start = orig_times.min().floor(freq)
    end = orig_times.max().ceil(freq)
    full_index = pd.date_range(start=start, end=end, freq=freq)

    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        if 'time' not in da.dims:
            new_vars[var] = da
            continue

        if np.issubdtype(da.dtype, np.number):
            if numeric_agg == 'mean':
                try:
                    da_grp = da.groupby('time_rounded').mean(dim='time', skipna=True)
                except TypeError:
                    da_grp = da.groupby('time_rounded').mean(dim='time')
            elif numeric_agg == 'first':
                da_grp = da.groupby('time_rounded').first()
            elif numeric_agg == 'last':
                da_grp = da.groupby('time_rounded').last()
            else:
                raise ValueError("numeric_agg must be 'mean', 'first', or 'last'")
        else:
            da_grp = da.groupby('time_rounded').first()

        da_grp = da_grp.rename({'time_rounded': 'time'})
        da_reg = da_grp.reindex(time=full_index)
        da_reg.attrs = da.attrs
        new_vars[var] = da_reg

    ds_reg = xr.Dataset(new_vars).assign_coords(time=full_index)

    # Copy over any non-time coords
    for c in ds.coords:
        if c in ('time', 'time_rounded'):
            continue
        if 'time' not in ds.coords[c].dims:
            ds_reg = ds_reg.assign_coords({c: ds.coords[c]})

    ds_reg.attrs = ds.attrs

    # If input was a DataArray, convert back
    if is_dataarray:
        da_reg = ds_reg['__temp__']
        da_reg.attrs = obj.attrs
        return da_reg

    return ds_reg