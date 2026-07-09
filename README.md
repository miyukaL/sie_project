# Sanity check of one year MRR data # 

Data must be in NetCDF format. 
Make run the sanity_check.py code to produce a csv file with all results of all tests for each hourly file in the dataset. 

The different checking are in the sanity_check_operations.py file and contain : 
# - find_zea_above_echotop : if there is values above the known echotop
# - check_single_timestamp : the number of timestamps in the file, and if the file is made of only one timestamp
# - check_single_timestamp_anf : same as before but only accounting for datapoints above the known near field artifact height
# - check_all_nan : if the file is all nan values
# - weird_timestamps : if there is unexpected timesteps, the percentage of those timesteps in the file, and their values
# - check_zea_range : the number and percentage of values aberrant for Zea reflectivity
# - check_nearfield_artefact : fraction of timestamps contaminated with near field artifact, and if this fraction is high enough, if the file is considered contaminated or not
# - check_truncated : number of timestamps, the expected number, and if the number of timestamps correspond to the expected one
# - check_separated_hours : if the hourly file is separated in several files
# - check_var_coherence : number of datapoints where there is a value for Z but not for Zea and the percentage of those points among all
# - check_laplacian : number of points with aberrant laplacian, and the percentage of those points among all
# - check_isolated_detections : number of isolated data points, and the percentage of those points among all
# - check_isolated_detections_noise : minimum reflectivity value considered, the number of isolated data points below this value, and the percentage of those points among all
