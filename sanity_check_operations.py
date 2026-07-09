import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from datetime import datetime
from scipy.ndimage import uniform_filter
from tqdm import tqdm
from joblib import Parallel, delayed

from regularized_time_grid import regularize_time_grid

#number of gates to exclude due to the near-field artefact#
NEAR_FIELD_AV = 4
NEAR_FIELD_AP = 6
CSV_PATH = "sanity_check_output/sanity_check.csv"


def file_exist_csv(folder, start_date, end_date):
    """Check if files exist for all expected hours between start_date and end_date."""
    dates = pd.date_range(start=start_date, end=end_date, freq="h")

    ds = []
    for dt in dates:
        date_str = f"{dt.year}/{dt.month:02d}/{dt.day:02d}"
        filename = f"{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}0000.nc"
        path = os.path.join(folder, date_str, filename)
        exists = os.path.isfile(path)
        ds.append({
            "date": f"{date_str}_{dt.hour:02d}:00",
            "file_path": path if exists else np.nan,
            "file_name": filename if exists else np.nan,
            "is_file_available": exists
        })

    df = pd.DataFrame(ds)
    return df


def find_zea_above_echotop(f, date='2025-05-26', echotop1=4445, echotop2=6375):
    ds = xr.open_dataset(f)
    
    if ds.time.values[0] < np.datetime64(date):
        echotop_threshold = echotop1
    else:
        echotop_threshold = echotop2
    
    above_zea_mask = ds.Zea.notnull() & (ds['range'] > echotop_threshold)
    valid_zea_mask = ds.Zea.notnull() & (ds['range'] <= echotop_threshold)
    
    n_aberrant = int(above_zea_mask.sum().compute())
    n_valide = int(valid_zea_mask.sum().compute())
    
    return pd.Series({
        'file': f.name,
        'n_above_echotop': n_aberrant,
        'n_valid_zea': n_valide,        # <-- ligne ajoutée
        'has_above_echotop': n_aberrant > 0
    })


def check_single_timestamp(f):

    "Count the number of files with only one timestamp"
    "f : file path"

    ds = xr.open_dataset(f)
    n_times = ds.sizes['time']
    
    return pd.Series({
        'file': f.name,
        'n_timestamps': n_times,
        'is_single_timestamp': n_times == 1
    })


def check_single_timestamp_anf(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26'):

    "Count the number of files with only one timestamp, and Zea values above near field"
    "n_gates : number of gates to ignore due to the neafield artefact"

    ds = xr.open_dataset(f)
    #Exclude the firsts gates due to neaf field artefact"
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds.Zea.isel(range=slice(n_gates, None))
    n_times = ds.sizes['time']

    return pd.Series({
        'file': f.name,
        'is_single_timestamp_anfa': n_times == 1 if zea.notnull().any() else np.nan  
    })


def check_all_nan(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26'):

    "Check if all Zea values are NaN above the 4th range"
    "f : file path"
    "n_gates : number of gates to ignore due to the neafield artefact"

    ds = xr.open_dataset(f)
    # Exclude the first gates due to near-field artefact
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds.Zea.isel(range=slice(n_gates, None))
    
    all_nan = bool(zea.isnull().all().compute())
    n_nonnan = int(zea.notnull().sum().compute())
    
    return pd.Series({
        'file': f.name,
        'all_nan': all_nan
    })


def weird_timestamps(f, atol=0.2, date='2025-05-26', expected_dt_before=10.0, expected_dt_after=5.0):

    "Check if timestamps are not regularly spaced"
    "f : file path"
    "atol : tolerance in seconds to account for small variations in the timestamps"
    "date : date of the MRR configuration change"
    "expected_dt_before : expected timestep before configuration change"
    "expected_dt_after : expected timestep after configuration change"

    ds = xr.open_dataset(f)
    ds = regularize_time_grid(ds)
    
    # Calculate the time differences in seconds between consecutive timestamps
    dt = pd.Series(ds.time.values).diff().dt.total_seconds().dropna()

    #Timesteps distance changes the 2025/05/26 due to MRR configurations changes
    expected_dt = expected_dt_before if ds.time.values[0] < np.datetime64(date) else expected_dt_after
    
    # Atol = 0.2 : tolerance of 0.2s to account for small variations in the timestamps 
    # aberrant_dt if : modulo close to 0 or close to expected_dt (9.9%10 -> 9.9 : close to 10 ; 10.1%10 -> 0.1 : close to 0)
    aberrant_dt = dt[~np.isclose(dt % expected_dt, 0, atol) & 
                ~np.isclose(dt % expected_dt, expected_dt, atol)] 
 
    
    return pd.Series({
        'file': f.name,
        'n_weird_timesteps': len(aberrant_dt),
        'pct_weird_timesteps': len(aberrant_dt) / len(dt) * 100 if len(dt) > 0 else 0,
        'values_weird_timesteps': aberrant_dt.values  #  raw values
    })


def check_zea_range(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26', inf=-30, sup=50):

    "Check if Zea values are in a realistic range (inf to sup dBZ) above the n_gates range"
    "f : file path"
    "n_gates : number of gates to ignore due to the neafield artefact"
    "inf : minimum realistic value for Zea in dBZ"
    "sup : maximum realistic value for Zea in dBZ"

    ds = xr.open_dataset(f)
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds.Zea.isel(range=slice(n_gates, None))  
    
    detected = zea.notnull()
    aberrant = detected & ((zea < inf) | (zea > sup))
    
    n_detected = int(detected.sum().compute())
    n_aberrant = int(aberrant.sum().compute())
    
    return pd.Series({
        'file': f.name,
        'n_aberrant_val': n_aberrant,
        'pct_aberrant_range': (n_aberrant / n_detected * 100) if n_detected > 0 else 0
    })


def check_nearfield_artefact(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26', cont_th=0.1):

    "Check if there is a near-field artefact"
    "n_gates : number of gates to ignore due to the neafield artefact"
    "cont_th : threshold for the fraction of contaminated timesteps to consider the file as contaminated"

    ds = xr.open_dataset(f)
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    nearfield = ds.Zea.isel(range=slice(1, n_gates))
    all_valid = nearfield.notnull().all(dim='range').compute()
    frac_contaminated = float(all_valid.mean())  # fraction of contaminated timesteps
    
    return pd.Series({
        'file': f.name,
        'frac_contaminated_nfa': frac_contaminated,
        'is_contaminated_nfa': frac_contaminated > cont_th # threshold for contaminated fraction
    })


def check_truncated(f, date1='2025-05-26', exp_date1=360, date2='2025-07-15', exp_date2=720):

    "Check if the file is truncated during the period where files should cover the whole houd regardless of the detection"
    "f : file path"
    "date1 : date until which the number of timestamps should respect the value exp_date1"
    "date2 : date until which the number of timestamps should respect the value exp_date2"

    n = xr.open_dataset(f).sizes['time']
    t0 = xr.open_dataset(f).time.values[0]
    
    if t0 < np.datetime64(date1):
        expected = exp_date1
    elif t0 < np.datetime64(date2):
        expected = exp_date2
    else:
        expected = None  # résolution non définie
    
    return pd.Series({
        'file': f.name,
        'n_timesteps': n,
        'expected': expected,
        'is_truncated': (expected is not None) and (1 < n < expected)
    })
    

def find_separated_hours(files):

    "Find hours for which there are several files, which can indicate separated hours"
    "files : list of file paths"

    df = pd.DataFrame({
        "file": [f.name for f in files],
        "yearmonthday": [f.name.split("_")[0] for f in files],
        "hour": [f.name.split("_")[1][:2] for f in files],
    })
    
    # grouper par jour+heure et compter le nombre de fichiers
    grouped = df.groupby(["yearmonthday", "hour"]).size().reset_index(name="n_files")
    separated = grouped[grouped["n_files"] > 1]
    
    return separated  

def check_separated_hours(file, separated):
    
    "Check if there are files that are separated"
    "file : file path"
    "separated : DataFrame of separated hours"

    yearmonthday = file.name.split("_")[0]
    hour = file.name.split("_")[1][:2]
    
    is_sep = ((separated["yearmonthday"] == yearmonthday) & 
              (separated["hour"] == hour)).any()
    
    return pd.Series({
        'file': file.name,
        'is_separated_hour': bool(is_sep)
    })


def check_var_coherence(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26'):

    "Check if the Zea, VEL and WIDTH variables are coherent, i.e if they are always together (all 3 are NaN or all 3 are not NaN) above the n_gates range"
    "f : file path"
    "n_gates : number of gates to ignore due to the neafield artefact"

    ds = xr.open_dataset(f)
    ds = regularize_time_grid(ds)  
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds.Zea.isel(range=slice(n_gates, None))
    vel = ds.VEL.isel(range=slice(n_gates, None))
    width = ds.WIDTH.isel(range=slice(n_gates, None))
    
    n_incoherent = int((vel.notnull() & zea.isnull()).sum().compute())
    n_incoherent += int((width.notnull() & zea.isnull()).sum().compute())
    n_incoherent += int((zea.notnull() & vel.isnull()).sum().compute())
    n_total = int(max(zea.size, vel.size, width.size))
    
    return pd.Series({
        'file': f.name,
        'n_incoherent': n_incoherent,
        'pct_incoherent': (n_incoherent / n_total * 100) if n_total > 0 else 0
    })


def check_laplacian(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26', threshold_range=20, threshold_time=20):

    "Check if there are abrupt changes in Zea values in time and range (Laplacian) above the 4th range between existing reflectivity values"
    "f : file path"
    "n_gates : number of gates to ignore due to the neafield artefact"
    "threshold_range : threshold for the gradient in range to consider it as aberrant"
    "threshold_time : threshold for the gradient in time to consider it as aberrant"

    ds = xr.open_dataset(f)
    ds = regularize_time_grid(ds)
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds['Zea'].isel(range=slice(n_gates, None))

    # Vertical gradient
    dZea_range = zea.diff(dim='range') # zea[r+1] - zea[r]
    # Vertical Boolean mask : True if the two values used to calculate gradient are non-NAN
    # Size : (time, range-1)
    both_valid_range = (
        zea.isel(range=slice(None, -1)).notnull() & # [0, 1, 2, ..., R-2, R-1]
        zea.isel(range=slice(1, None)).notnull() # [1, 2, ..., R-2, R-1, R] 
    )

    # Temporal gradient
    dZea_time = zea.diff(dim='time') # zea[t+1] - zea[t]
    # Temporal Boolean mask : True if the two values used to calculate gradient are non-NAN
    # Size : (time-1, range)
    both_valid_time = (
        zea.isel(time=slice(None, -1)).notnull() & # [0, 1, 2, ..., T-2, T-1]
        zea.isel(time=slice(1, None)).notnull() # [1, 2, ..., T-2, T-1, T] 
    )

    # Align on same grid of size : (time-1, range-1)
    dZea_range_aligned = dZea_range.isel(time=slice(None, -1))
    dZea_time_aligned  = dZea_time.isel(range=slice(None, -1))
    both_valid_aligned = both_valid_range.isel(time=slice(None, -1)) & both_valid_time.isel(range=slice(None, -1))

    # Magnitude of the 2D gradient as euclidian norm √(threshold_range² + threshold_time²)
    gradient_2d = np.sqrt(dZea_range_aligned**2 + dZea_time_aligned**2) 

    # Aberrant if above threshold and datas are valid
    aberrant = (gradient_2d > np.sqrt(threshold_range**2 + threshold_time**2)) & both_valid_aligned 

    n_aberrant = int(aberrant.sum().compute())
    n_valid = int(both_valid_aligned.sum().compute())

    return pd.Series({
        'file': f.name,
        'n_aberrant_laplacian': n_aberrant,
        'pct_aberrant_laplacian': (n_aberrant / n_valid * 100) if n_valid > 0 else 0
    })


def check_isolated_detections(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26', radius_range=5, radius_time=5):

    "Check spikes of isolated reflectivity surronded by Nans"
    "f : file path"
    "n_gates : number of gates to ignore due to the near field artefact"
    "radius_range : number of gates to consider in the range dimension for the neighborhood"
    "radius_time : number of timesteps to consider in the time dimension for the neighborhood"

    ds = xr.open_dataset(f)
    ds = regularize_time_grid(ds)
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds['Zea'].isel(range=slice(n_gates, None)).compute()
    
    # Mask of non-NaN values (valid detections) : True where Zea is not NaN, False where Zea is NaN
    valid = zea.notnull().values  
    
    # Number of non-NaN values in the neighborhood (2*radius+1) x (2*radius+1)
    # uniform_filter gives the mean valus so need to multiply by window size to get number
    size = (2 * radius_time + 1, 2 * radius_range + 1)
    n_neighbors = uniform_filter(valid.astype(float), size=size, mode='constant', cval=0)
    n_neighbors = np.round(n_neighbors * size[0] * size[1]).astype(int)
    
    # Isolated if alone in neighborhood
    # -1 to not count the point itself
    isolated = valid & (n_neighbors - 1 < 1) 
    
    n_isolated = int(isolated.sum())
    n_valid = int(valid.sum())
    
    return pd.Series({
        'file': f.name,
        'n_isolated': n_isolated,
        'pct_isolated': (n_isolated / n_valid * 100) if n_valid > 0 else 0
    })


def check_isolated_detections_noise(f, n_gates_av=NEAR_FIELD_AV, n_gates_ap=NEAR_FIELD_AP, date='2025-05-26', radius_range=5, radius_time=5, z_min=-10 ):  

    "Check spikes of isolated reflectivity surronded by Nans, with a threshold on the reflectivity value to consider only the smallest reflectivity ones"
    "f : file path"
    "n_gates : number of gates to ignore due to the near field artefact"
    "radius_range : number of gates to consider in the range dimension for the neighborhood"
    "radius_time : number of timesteps to consider in the time dimension for the neighborhood"
    "z_min : minimum reflectivity value to consider"

    ds = xr.open_dataset(f)
    ds = regularize_time_grid(ds)
    n_gates = n_gates_av if ds.time.values[0] < np.datetime64(date) else n_gates_ap
    zea = ds["Zea"].isel(range=slice(n_gates, None)).compute()

    # Mask of non-NaN values (valid detections) : True where Zea is not NaN, False where Zea is NaN
    valid = zea.notnull()
    valid_arr = valid.values.astype(float)

    # weak reflectivity values only
    valid_below = valid & (zea < z_min)
    valid_below = valid_below.values.astype(bool)

    # Number of non-NaN values in the neighborhood (2*radius+1) x (2*radius+1)
    # uniform_filter gives the mean valus so need to multiply by window size to get number
    size = (2 * radius_time + 1, 2 * radius_range + 1)
    n_neighbors = uniform_filter(valid_arr, size=size, mode="constant", cval=0)
    n_neighbors = np.round( n_neighbors * size[0] * size[1]).astype(int)

    # Isolated if alone in neighborhood
    # -1 to not count the point itself
    isolated = valid_below & (n_neighbors - 1 == 0)

    n_isolated = int(isolated.sum())
    n_valid = int(valid.sum())

    pct = (n_isolated / n_valid * 100) if n_valid > 0 else np.nan

    return pd.Series(
        {
            "file": f.name,
            "z_min": z_min,
            "n_isolated_weak": n_isolated,
            "pct_isolated_weak": pct,
        }
    )


def add_to_csv(df_csv, df_results, csv_path):

    "Add the results of the function to the csv file as pandas dataframe"
    "df_csv : csv file to update as pandas dataframe"
    "df_results : results dataframe to add to the csv file"
    "csv_path : path to the csv file"

    df_csv.merge(df_results, left_on='file_name', right_on='file', how='left') \
          .drop(columns='file') \
          .to_csv(csv_path, index=False)


def run_check_if_needed(check_fn, files, columns, csv_path=CSV_PATH, n_jobs=8, **kwargs):

    "Check if columns to compute were already computed and added to the csv"
    "check_fn : function to compute the columns"
    "files : list of files to check"
    "columns : list of columns to compute"
    "csv_path : path of the csv created with file_exist_csv()"
    "n_jobes : number of parallel jobs to run the check function"

    # Open csv as pandas dataframe
    df = pd.read_csv(csv_path) 
    
    # Check if columns to compute already exist 
    if all(col in df.columns for col in columns):
        print(f"{columns} already computed, skipping.")
        return
    
    # If columns to compute don't exist : compute them in parallel and add to csv
    results = Parallel(n_jobs=n_jobs)(delayed(check_fn)(f, **kwargs) for f in tqdm(files))
    add_to_csv(df, pd.DataFrame(results), csv_path)
    print(f"Done. Columns {columns} added to {csv_path}.")
    return 



