"""
Script for raw MRR data sanity check.
 
Generate or complete a CSV file listing, for each expected .nc file, 
its presence and a series of diagnostics (echotop, temporal isolation, 
aberrant values, artifacts, etc.).
 
Usage:
    python run_sanity_check.py
"""

import os
from pathlib import Path   
from datetime import datetime 
from sanity_check_operations import (
    file_exist_csv,
    find_zea_above_echotop,
    check_single_timestamp,
    check_single_timestamp_anf,
    check_all_nan,
    check_zea_range,
    check_nearfield_artefact,
    check_truncated,
    check_separated_hours,
    find_separated_hours,
    check_regularized_all,
    run_check_if_needed,
)
# --------------------------------------------------------------------------------------------------
# CONFIGURATION 
# --------------------------------------------------------------------------------------------------

FOLDER = "/awaca/ddu/mrr/data/raw_data/"
YEAR = 2025
OUTPUT_DIR = "sanity_check_output"
CSV_PATH = os.path.join(OUTPUT_DIR, "sanity_check.csv")

# --------------------------------------------------------------------------------------
# FUNCTIONS SUMMARY
# - find_zea_above_echotop : ['n_above_echotop', 'n_valid_zea', 'has_above_echotop']
# - check_single_timestamp : ['n_timestamps', 'is_single_timestamp']
# - check_single_timestamp_anf : ['is_single_timestamp_anfa']
# - check_all_nan : ['all_nan']
# - check_zea_range : ['n_aberrant_val', 'pct_aberrant_range']
# - check_nearfield_artefact : ['frac_contaminated_nfa', 'is_contaminated_nfa']
# - check_truncated : ['n_timesteps', 'expected', 'is_truncated']
# - check_separated_hours : ['is_separated_hour']
# - check_regularized_all : ['n_weird_timesteps', 'pct_weird_timesteps', 'values_weird_timesteps',
#                             'n_incoherent', 'pct_incoherent',
#                             'n_aberrant_laplacian', 'pct_aberrant_laplacian',
#                             'n_isolated', 'pct_isolated',
#                             'z_min', 'n_isolated_weak', 'pct_isolated_weak']
# -----------------------------------------------------------------------------------------

def ensure_base_csv(folder: str, year: int, csv_path: str) -> None:
    """Create the base CSV (presence of files) if it doesn't already exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) 
 
    if not os.path.exists(csv_path):
        df = file_exist_csv(
            folder, 
            start_date=datetime(year, 1, 1, 0), 
            end_date=datetime(year, 12, 31, 23),
            )
        df.to_csv(csv_path, index=False)
 
 
def run_all_checks(files, csv_path: str) -> None:
    """Run all sanity checks on the given list of files and update the CSV file with the results."""
 
    # ECHOTOP
    run_check_if_needed(
        find_zea_above_echotop, files,
        ['n_above_echotop', 'n_valid_zea', 'has_above_echotop'],
        csv_path=csv_path,
    )
 
    # ISOLATION IN TIME
    run_check_if_needed(
        check_single_timestamp, files,
        ['n_timestamps', 'is_single_timestamp'],
        csv_path=csv_path,
    )
 
    # ... only above the near field artifact (ANFA)
    run_check_if_needed(
        check_single_timestamp_anf, files,
        ['is_single_timestamp_anfa'],
        csv_path=csv_path,
    )
 
    # FILES WITH ONLY NAN VALUES
    run_check_if_needed(
        check_all_nan, files,
        ["all_nan"],
        csv_path=csv_path,
    )
 
    # ABERRANT ZEA VALUES
    run_check_if_needed(
        check_zea_range, files,
        ['n_aberrant_val', 'pct_aberrant_range'],
        csv_path=csv_path,
    )
 
    # FILES WITH NEARFIELD ARTEFACT
    run_check_if_needed(
        check_nearfield_artefact, files,
        ['frac_contaminated_nfa', 'is_contaminated_nfa'],
        csv_path=csv_path,
    )
 
    # TRUNCATED FILES
    run_check_if_needed(
        check_truncated, files,
        ['n_timesteps', 'expected', 'is_truncated'],
        csv_path=csv_path,
    )
 
    # SEPARATED HOURS
    separated = find_separated_hours(files)
    run_check_if_needed(
        check_separated_hours, files,
        ['is_separated_hour'],
        csv_path=csv_path,
        separated=separated,
    )
 
    # ALL CHECKS THAT NEED THE REGULARIZED TIME GRID, IN ONE PASS
    # (weird timestamps, variable coherence, laplacian, isolated detections,
    # isolated weak detections) -> regularize_time_grid() runs once per file.
    run_check_if_needed(
        check_regularized_all, files,
        ['n_weird_timesteps', 'pct_weird_timesteps', 'values_weird_timesteps',
         'n_incoherent', 'pct_incoherent',
         'n_aberrant_laplacian', 'pct_aberrant_laplacian',
         'n_isolated', 'pct_isolated',
         'z_min', 'n_isolated_weak', 'pct_isolated_weak'],
        csv_path=csv_path,
    )

   
 
 
def main() -> None:
    ensure_base_csv(FOLDER, YEAR, CSV_PATH)
 
    root = Path(FOLDER) / str(YEAR)
    files = sorted(root.rglob("*.nc"))
 
    run_all_checks(files, CSV_PATH)
 
 
if __name__ == "__main__":
    main()