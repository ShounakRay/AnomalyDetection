############################################################
#################### DEPENDENCY IMPORTS ####################
# For file IO
import sys
import os
import io
# For processing
import pandas as pd
os.chdir("/Users/Ray/Documents/Python/9 - Oil and Gas")
sys.path.append("/Users/Ray/Documents/Python/8 - Data Preparation")
sys.path.append("/Users/Ray/Documents/Python/REFERENCE FILES/distfit-1.1.5/distfit/utils")
from generic_data_prep import *
from generic_data_modeling import *
from smoothline import *
import datetime
from datetime import datetime, timedelta
# For analytics
import random
import numpy as np
from numpy import diff
import itertools
from itertools import chain
import sklearn
from sklearn.ensemble import IsolationForest
import ruptures as rpt
# import rpy2
# from rpy2.robjects import FloatVector
# from rpy2.robjects.packages import importr
# utils = importr('utils')
# utils.chooseCRANmirror(ind = 1)
# r_packages = ['cpm', 'EnvCpt', 'stats']
# for package in r_packages:
#     utils.install_packages(package)
#     exec(package + " = importr('" + package + "')")
# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
# For saving
import pickle
import json
from collections import Counter as Counter

############################################################
##################### HYPER PARAMETERS #####################
__FOLDER__ = r'/Users/Ray/Documents/Python/9 - Oil and Gas'
PATH_SANDALL = __FOLDER__ + r'/Data/Sandall/sandall.csv'
PATH_EDAM_EAST = __FOLDER__ + r'/Data/Edam/edam_east.csv'
PATH_EDAM_WEST = __FOLDER__ + r'/Data/Edam/edam_west.csv'
PATH_VAWN = __FOLDER__ + r'/Data/Vawn/vawn_pvr_daily_2020_03_09.csv'

INTERVAL = 90 # days, moving window width
STEP = 15 # days, window translation amount per iteration
PAD_COL = 'pad_name'
GROUPBY_COL = 'pair_name'
TIME_COL = 'production_date'

### MAIN HARVESTER
# Iterate through all windows for outlier detection, statistical recording, etc....
def feature_harvest(time_ranges, filtered_df, foi, time_col, groupby_col, groupby_value):
    well_statistics = []
    START_TIME = datetime.now()
    for t_range in time_ranges:
        # Filter DataFrame further for given window
        window_filtered_df = reset_df_index(filtered_df[(filtered_df[time_col] > t_range[0]) & (filtered_df[time_col] < t_range[1]) & (filtered_df[groupby_col] == groupby_value)])
        title = (groupby_value + ": " + str(t_range[0]) + " to " + str(t_range[1])).replace(' 00:00:00', '')

        # Proceed forward if original section is empty
        if(window_filtered_df.empty):
            process_logger("WINDOW EMPTY: " + str(t_range[0]) + " - " + str(t_range[1]), START_TIME, name = 'feature_script_logger.txt')
            continue

        # Determine outliers for given window
        outliers, trimmed, pct_out = outlier_detection(window_filtered_df, foi, title, manual_contamination = 0.1, plot = False)

        # Proceed forward if trimmed section is empty
        if(trimmed.empty):
            # If original window is not empty, then `outlier_detection` flagged all points as outliers. Likely wrong, so treat original as 100% valid.
            if not(window_filtered_df.empty):
                trimmed = window_filtered_df
            else:
                process_logger("TRIMMED WINDOW ↓ EMPTY: " + str(t_range[0]) + " - " + str(t_range[1]), START_TIME, name = 'anomaly_script_logger.txt')
                continue

        # >> Application: Determine statistics for current windowed distribution and store
        univ_stats = statistics(trimmed, foi)
        # Additional modifications to add/compress information to a single DataFrame
        univ_stats['window_start'] = t_range[0]
        univ_stats['window_end'] = t_range[1]
        univ_stats['percent_outliers'] = pct_out

        well_statistics.append(tuple(univ_stats.iloc[0]))
        process_logger(groupby_value + ", " + foi + ": " + str(t_range[0]) + " - " + str(t_range[1]), START_TIME, name = 'feature_script_logger.txt')

    # There was absolutely no (trimmed) data to scrape in the definied windows, then:
    if(len(well_statistics) == 0):
        return [('EMPTY')], ['EMPTY']
    else:
        stat_features = list(univ_stats.columns)
    return well_statistics, stat_features
# Master Function (INCOMPLETE)
def find_anomalies(recording, well_name):
    minimas = recording[well_name]

### ANOMALY DETECTION METHODS
# Method A: Univariate Outlier Detection [Unsupervised Anomaly Detection] Process
def outlier_detection(df, foi, title, y_label = 'Frequency', FIG_SIZE = (11.7, 8.27), manual_contamination = None, plot = False):
    if(manual_contamination is None):
        forest = IsolationForest(n_estimators = 100, contamination = 'auto', bootstrap = False)
    elif(manual_contamination > 0.0 and manual_contamination < 1.0):
        forest = IsolationForest(n_estimators = 100, contamination = manual_contamination, bootstrap = False, random_state = 303928462)
    else:
        raise ValueError("'manual_contamination' argument is out-of-bounds.")

    if(foi == None):
        foi = df.columns[0]
        y_label = 'Value'

    forest.fit(df[foi].values.reshape(-1, 1))
    reshaped_x = np.array(list(set(list(np.sort(df[foi].values))))).reshape(-1, 1) # np.linspace(df[foi].min(), df[foi].max(), len(df)).reshape(-1, 1)
    anomaly_score = forest.decision_function(reshaped_x)
    outlier = forest.predict(reshaped_x)

    if(plot):
        plot_anomaly_scores(reshaped_x, anomaly_score, 'Anomaly Score', outlier, title, foi, FIG_SIZE = FIG_SIZE)
        plot_outliers(reshaped_x, df, foi, title, outlier, bins = 100, FIG_SIZE = FIG_SIZE)

    # Zipped Comprehension: 0.009
    temp = list(zip([i[0] for i in reshaped_x], outlier))
    outliers = [tup[0] for tup in temp if(tup[1] == -1)]
    trimmed = [tup[0] for tup in temp if(tup[1] == 1)]

    outlier_pct = str(round((len(outliers)/float(len(temp)))*100, 3))
    # print('% Outliers Detected: ' + outlier_pct + "%")

    outliers_ret = reset_df_index(df[df[foi].map(lambda x: x in outliers)])
    trimmed_ret = reset_df_index(df[df[foi].map(lambda x: x in trimmed)])

    return outliers_ret, trimmed_ret, float(outlier_pct)
# >> Plots anomaly scores for IsolationForest instance predictions
def plot_anomaly_scores(reshaped_x, y, y_label, outlier, title, foi, FIG_SIZE = (11.7, 8.27)):
    plt.figure(figsize = FIG_SIZE)
    plt.plot([i[0] for i in reshaped_x], y, label = y_label)
    plt.fill_between(reshaped_x.T[0], np.min(y), np.max(y),
                     where = outlier == -1, color = 'r',
                     alpha = 0.4, label ='Outlier Region')
    plt.legend()
    plt.suptitle(title)
    plt.ylabel('Anomaly Score')
    plt.xlabel(foi)
    plt.show()
# >> Plots outliers in original density plot / distribution
def plot_outliers(reshaped_x, df, foi, title, outlier, bins = 100, FIG_SIZE = (11.7, 8.27)):
    sns.set(rc = {'figure.figsize': FIG_SIZE})
    ax = sns.distplot(df[foi], bins = bins, color = 'b')
    y_data = [rect.get_height() for rect in sns.distplot((df[foi]), bins = 100).patches]
    ax.set_title(title)
    ax.set_ylabel('Probability Density')
    ax.fill_between(reshaped_x.T[0], min(y_data), max(y_data),
                     where = outlier == -1, color = 'r',
                     alpha = 0.4, label = 'Outlier Region')
    plt.show()

# Method B: Univariate Statistical Fetures
def statistics(df, foi, detailed = True):
    stats, _ = stat_metrics(df, foi, detailed = True)
    return stats
# >> Plots the frequency distribution and time-series for the metric of interest
def plot_stats(df, metric, FIG_SIZE = (11.7/2, 8.27/2)):
    plt.figure(figsize = FIG_SIZE)
    plt.plot(list(df.index), df[metric], label = metric)
    plt.ylabel(metric)
    plt.xlabel('Time interval')
    plt.legend()
    plt.suptitle('Progression of \"' + metric + '\" metric over time interval', fontsize = 12)
    plt.show()
    plt.close()
    sns.set(rc = {'figure.figsize': tuple(0.952 * dim for dim in FIG_SIZE)})
    ax = sns.distplot(df[metric], bins = 20)
    ax.set_title('Frequency of \"' + metric + '\" in total time interval.', fontsize = 12)
    ax.set_ylabel('Frequency')
    plt.show()
    plt.close()

# Method C: Offline Change Detection
## For gradual, nuanced statistical data
### Note: mean/median approximates "value" in time-series application if INTERVAL << 0
def stat_change_detection(harvest, group, feature, metric):
    metric_data = harvest[group][feature][metric]
## For time_series data
# Performs offline change detection on time-series data
def change_detection_ts(data, penalty = 10, model = 'rbf', plot = True, title_append = '', FIG_SIZE = (12.0, 8.27)):
    title_append = "\n" + title_append
    algo = rpt.Pelt(model = "rbf").fit(data)
    if(penalty == 'bic'):
        T = np.asanyarray(data).shape
        sigma = np.asanyarray(data).std()
        bic_mod = np.sqrt(sigma) * np.log(T) * 1.0
        penalty = bic_mod
    result = algo.predict(pen = penalty)
    # display
    # rpt.display(time_series, [0], result)
    if(plot):
        figure = plt.figure(figsize = FIG_SIZE)
        plt.plot([x for x in range(len(data))], data)
        for lines in result:
            plt.axvline(x = lines, color = 'black', dashes = (2, 4))
        plt.title('Offline Change Detection' + title_append)
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.show()
        plt.close()
    return result

### UTILITY FUNCTION
# Calculate all windows given start, end, and interval (optional last date range fitting disabled, deletion enabled)
def windows(t_MIN, t_MAX, interval, moving = True, step = 1):
    if(moving):
        all = [(t_MIN + timedelta(days = step * iteration), t_MIN + timedelta(days = step * iteration + interval), iteration + 1) for iteration in range(round((t_MAX - t_MIN).days/step))]
    else:
        all = [(t_MIN + timedelta(days = iteration * interval), t_MIN + timedelta(days = (iteration + 1) * interval), iteration + 1) for iteration in range(round((t_MAX - t_MIN).days/interval))]
    while(all[-1][1] > t_MAX):
        all.pop()
        # all[-1] = (all[-1][0], t_MAX)
    return all
# Check match of specified feature across two DataFrames
def crossmatch(df_1, df_2, feature):
    if(len(df_1) != len(df_2)):
        return False
    bool_array = list((df_1[feature] == df_2[feature]).unique())
    if(True in bool_array):
        return True
    return False
# Plot extremas in data
def extremas(dist, FIG_SIZE = (11.7, 8.27), plot = False, title = '', normalize_y = True):
    min_indices = localmins(dist)[0]
    max_indices = scipy.signal.find_peaks(dist)[0]
    if(plot):
        fig = plt.figure(figsize = FIG_SIZE)
        plt.plot([x for x in range(len(dist))], dist)
        for mark in min_indices:
            plt.axvline(x = mark, color = 'blue', dashes = (2, 4))
        for mark in max_indices:
            plt.axvline(x = mark, color = 'red', dashes = (2, 4))
    min_info = dict([(min_index, np.abs(dist[min_index])) for min_index in min_indices])
    max_info = dict([(max_index, dist[max_index]) for max_index in max_indices])

    min_norm = {}
    max_norm = {}

    if(normalize_y):
        extremas = list(min_info.values())
        min_norms = util_normalize(extremas)
        for i in range(len(min_info)):
            min_norm[extremas[i]] = min_norms[i]
        extremas = list(max_info.values())
        max_norms = util_normalize(extremas)
        for i in range(len(max_info)):
            max_norm[extremas[i]] = max_norms[i]

    return min_info, max_info
# Find all local minimas
def localmins(a):
    a = np.array(a)
    begin = np.empty(a.size//2+1,np.int32)
    end = np.empty(a.size//2+1,np.int32)
    i = k = 0
    begin[k]=0
    search_end=True
    while i < a.size-1:
        if a[i]>a[i+1]:
            begin[k]=i+1
            search_end=True
        if search_end and a[i]<a[i+1]:
            end[k]=i
            k+=1
            search_end=False
        i += 1
    if search_end and i>0  : # Final plate if exists
        end[k]=i
        k+=1
    return begin[:k],end[:k]

    print(*zip(*localmins(test03)))
# Ensure selected keys actaully exist in DataFrame
def verify_params(df, specified, actual):
    alert = False
    for val_expected in specified:
        if(val_expected not in actual):
            print('Value "' + val_expected + '" not present inputted data.')
            alert = True
    if(alert):
        raise ValueError('One or more incompatibilities in specified and actual elements.')
# Major process logger
def process_logger(record, START_TIME, name = 'script_logger.txt'):
    if not os.path.exists(name):
        text_file = open(name, "w")
    else:
        text_file = open(name, "a+")

    elapsed_time = datetime.now() - START_TIME
    last_time = START_TIME + elapsed_time
    delta_time = datetime.now() - last_time

    text_file.writelines('Elapsed: ' + str(elapsed_time) + "\t →  " + record + '\n')
    # text_file.writelines('Elapsed: ' + str(elapsed_time) + "\t Delta: " + str(delta_time) + ":: " + record + '\n')
    text_file.close()
# Visualization of ~space compexity for `windows` algorithm
def analyze_window_algo(t_EXTREMES):
    a = []
    color_map = []
    for j in range(1, 100):
        for i in range(1, 1000):
            frames = windows(t_EXTREMES[0], t_EXTREMES[1], interval = i, step = j)
            window_size = len(frames)
            if(frames[-1][1] < t_MAX):
                color_map.append('red')
            else:
                color_map.append('blue')
            a.append((j, i, window_size))

    fig = plt.figure(figsize = (44, 32))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('step size')
    ax.set_ylabel('interval size')
    ax.set_zlabel('window size')
    ax.scatter([i[0] for i in a], [i[1] for i in a], [i[2] for i in a], c = color_map)
    plt.show()
# Function to convert an R object to a Python dictionary
def robj_to_dict(robj):
    return dict(zip(robj.names, map(list, robj)))
# Pickle object
def save_pickle(name, data):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)
# Load picked object
def load_pickle(name):
    with open(name + '.pickle', 'rb') as handle:
        pickled = pickle.load(handle)
    return pickled
# Merge dicts
def merge_dicts(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]
    return d

############################################################
##################### DATA PROCESSING ######################
df_sandall = pd.read_csv(PATH_SANDALL)
df_edamE = pd.read_csv(PATH_EDAM_EAST)
df_edamW = pd.read_csv(PATH_EDAM_WEST)
df_vawn = pd.read_csv(PATH_VAWN)

df_ALL = [df_sandall, df_edamE, df_edamW, df_vawn]

# Updating the elements in a list inplace will mutate the original
_ = [df.drop('Unnamed: 0', 1, inplace = True) for df in df_ALL if 'Unnamed: 0' in df]

############################################################
#################### ANOMALY DETECTION #####################
########################################
######### PARAMETER SETTINGS ###########
df = df_sandall.copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df[TIME_COL] = df[TIME_COL].apply(lambda x: x.date())

# Determine all unique stratification groups and features
ALL_GROUPS = df[GROUPBY_COL].unique()
# Also linked to MODEL INPUT !!!!!
ALL_FEATURES = [
 'dly_stm',
 'inj_tubing_pressure',
 'inj_casing_pressure',
 'stm_tubing_temperature',
 'chlorides',
 'oil_sales',
 'water_sales',
 'gas_sales',
 'prd_tubing_pressure',
 'prd_casing_pressure',
 'prd_tubing_temperature',
 'spm_rpm',
 'pump_efficiency',
 # 'prod_runtime_hours',
 # 'inj_runtime_hours'
 'runtime_hours'
]

# Verify that all group and feature paramaters exist in dataset
verify_params(df, specified = ALL_FEATURES, actual = df.columns)
verify_params(df, specified = ALL_GROUPS, actual = df[GROUPBY_COL].unique())

########################################
######### FEATURE ENGINEERING ##########
# Iterate through all features and wells to retrieve statistical data
STATS_by_group = {}
STATS_by_feature = {}
for group in ALL_GROUPS:
    START_TIME = datetime.now()
    print('> Group: ' + group)
    groupby_value = group
    for feature in ALL_FEATURES:
        print('>> Feature: ' + feature)
        # Specifiy settings for this iteration
        foi = feature

        # Determine filtered DataFrame based on chosen parameters; and
        # Sort DataFrame by time
        filtered_df = reset_df_index(df[df[GROUPBY_COL] == groupby_value].sort_values(by = TIME_COL))
        # Determine start and end days for DataFrame
        t_MIN = filtered_df[TIME_COL][0]
        t_MAX = filtered_df[TIME_COL][filtered_df.shape[0] - 1]
        t_EXTREMES = (t_MIN, t_MAX)

        # Determine window ranges
        time_ranges = windows(t_EXTREMES[0], t_EXTREMES[1], interval = INTERVAL, step = STEP)
        # Harvests statistical features from outlier-trimmed dataset
        well_statistics, headers = feature_harvest(time_ranges, filtered_df, foi, TIME_COL, GROUPBY_COL, groupby_value)
        # Assign and update statistical information to feature tracker
        well_stats = pd.DataFrame(well_statistics, columns = headers)
        STATS_by_feature[feature] = well_stats.copy()

    # Update statistical information to group tracker
    STATS_by_group[group] = STATS_by_feature.copy()
    STATS_by_feature.clear()
    process_logger(group, START_TIME, name = 'group_script_logger.txt')

### SAVE DATA ###
name = 'sandall_harvest_outliers-statistics_higheres'
# Pickle (storage purposes) and open (test purposes) Harvest
# save_pickle(name, STATS_by_group)
STATS_by_group = load_pickle(name)

##############################################################
################### POST-PROCESSING ANALYTICS ################
# Check depth of loaded/determined information
for group in STATS_by_group.keys():
    print('>>> For "' + group + '": ')
    for feature in STATS_by_group[group].keys():
        print('\t> Calculated "' + feature + '" ')
        stats_data = STATS_by_group[group][feature]

# Time-Series Change Detection
groupby_value = 'SA1_SA2'
feature = 'dly_stm'
time_series = np.array(df[df[GROUPBY_COL] == groupby_value][feature])
# R Code
"""_r = {
# time_series = random.sample(time_series, int(0.5 * len(time_series)))
# cpm_result = cpm.detectChangePoint(FloatVector(time_series), cpmType = 'Mann-Whitney', ARL0 = 100, startup = 100)
# cpm_result = robj_to_dict(cpm_result)
# time = cpm_result.get('detectionTime')[0]

# fit_envcpt = EnvCpt.envcpt(FloatVector(time_series))
# envcpt_result = robj_to_dict(fit_envcpt)
}"""

#############################################################
#############################################################
# Time-series, NOT statistical
# Establishing Confidence in Pad-Specific, Feature-Specific Changepoints (Offline)
changepoint_record_by_pad = {}
changepoint_record_by_group = {}
changepoint_record_by_feature = {}
for pad in df[PAD_COL].unique():
    df_filtered = df[df[PAD_COL] == pad]
    for group in df_filtered[GROUPBY_COL].unique():
        for feature in ALL_FEATURES:
            time_series = np.array(df_filtered[df_filtered[GROUPBY_COL] == group][feature])
            changepoints = change_detection_ts(time_series, penalty = 'bic', plot = False, title_append = pad + ", " + group + ", " + feature)
            changepoint_record_by_feature[feature] = changepoints
        changepoint_record_by_group[group] = changepoint_record_by_feature.copy()
        changepoint_record_by_feature.clear()
    changepoint_record_by_pad[pad] = changepoint_record_by_group.copy()
    changepoint_record_by_group.clear()
    print(pad + " processing completed.")

### SAVE DATA ###
# save_pickle('changepoint_record_by_pad', changepoint_record_by_pad)
changepoint_record_by_pad = load_pickle('changepoint_record_by_pad')

########################################
###### ONLINE ANOMALY DETECTION ########
# Local Dependencies: reset_df_index, snakify (2) functions,
# Python Dependencies: IsolationForest, itertools
def phase_outlier_detection(data, well, feature, mode, diff_thresh = 100, net_contamination = 0.1, phase_contamination = 0.01, TIME_COL = 'production_date', PAD_COL = 'pad_name', GROUPBY_COL = 'pair_name', plot = False):
    # Snakify columns and feature name
    data = util_snakify_cols(data)
    feature, TIME_COL, PAD_COL, GROUPBY_COL = snakecase(feature).replace('__', '_'), snakecase(TIME_COL).replace('__', '_'), snakecase(PAD_COL).replace('__', '_'), snakecase(GROUPBY_COL).replace('__', '_')

    # Data-type verification and variable settings
    FIG_SIZE = (12, 8.27)
    data = pd.DataFrame(data)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data[TIME_COL] = data[TIME_COL].apply(lambda x: x.date())
    pad = reset_df_index(data[data[GROUPBY_COL] == well])[PAD_COL][0]
    well = str(well)
    feature = str(feature)
    if(mode not in ['changepoint', 'overall']):
        raise ValueError('XXX `mode` wrong input. XXX')
    elif(mode == 'changepoint'):
        net_contamination = None
        if(phase_contamination != 'auto'):
            phase_contamination = float(phase_contamination)
            if(phase_contamination < 0.0 or phase_contamination > 1.0):
                raise ValueError('XXX `phase_contamination` is outside 0-1 range. XXX')
    elif(mode == 'overall'):
        phase_contamination = None
        if(net_contamination != 'auto'):
            net_contamination = float(net_contamination)
            if(net_contamination < 0.0 or net_contamination > 1.0):
                raise ValueError('XXX `net_contamination` is outside 0-1 range. XXX')
    diff_thresh = int(diff_thresh)
    TIME_COL = str(TIME_COL)
    ALL_FEATURES = [
     'dly_stm',
     'inj_tubing_pressure',
     'inj_casing_pressure',
     'stm_tubing_temperature',
     'chlorides',
     'oil_sales',
     'water_sales',
     'gas_sales',
     'prd_tubing_pressure',
     'prd_casing_pressure',
     'prd_tubing_temperature',
     'spm_rpm',
     'pump_efficiency',
     # 'prod_runtime_hours',
     # 'inj_runtime_hours'
     'runtime_hours'
    ]

    # High-level data re-structuring
    data = reset_df_index(data[(data[PAD_COL] == pad) & (data[GROUPBY_COL] == well)]).sort_values(by = TIME_COL)

    normalized_feature_data = data.copy()[ALL_FEATURES + [TIME_COL, PAD_COL, GROUPBY_COL]]
    for feature in ALL_FEATURES:
        normalized_feature_data[feature] = util_normalize(normalized_feature_data[feature])

    data = data[[feature, TIME_COL, PAD_COL, GROUPBY_COL]]
    # !!! NOTE THESE COLUMN NAMES MUST BE SNAKE_CASE COMPATIBLE
    data['anomaly'] = 'No'
    data['score'] = None

    # Analyze whole dataset for outlier detection
    if(mode == 'overall'):
        ### Full-based Outlier Detection
        clf = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = net_contamination, max_features = 1.0, behaviour = 'new')
        clf.fit(data[[feature]])
        info = clf.decision_function(data[[feature]])
        anomalies = clf.predict(data[[feature]])
        if(plot):
            fig, ax = plt.subplots(figsize = FIG_SIZE)
            plt.plot(data[TIME_COL], data[feature])
        for status_i in range(len(anomalies)):
            if(anomalies[status_i] == -1):
                dpt = data[feature][status_i]
                data.at[status_i, 'anomaly'] = 'Yes'
                data.at[status_i, 'score'] = info[status_i]
                if(plot):
                    ax.scatter(data[TIME_COL][status_i], dpt, facecolors = 'none', edgecolors = 'r')
        if(plot):
            plt.show()
    # Analyze dataset for outlier detection in segments
    else:
        ### Phase-based Outlier Detection
        diff_thresh = diff_thresh
        min_time = data[TIME_COL].iloc[0]
        max_time = data[TIME_COL].iloc[-1]
        new_sections = []
        all_groups = []
        now_time = min_time

        # Find change points and determine windows
        cpoints = rpt.Pelt(model = "rbf").fit(np.array(data[feature])).predict(pen = 10)
        cpoints = [cpoints[i] for i in range(len(cpoints) - 1) if cpoints[i + 1] >= cpoints[i] + diff_thresh]
        if((max_time - min_time).days + 1 not in cpoints):
            cpoints.insert(len(cpoints), (max_time - min_time).days + 1)
        for i in range(len(cpoints)):
            new_sections.append((now_time, min_time + timedelta(days = cpoints[i])))
            now_time = min_time + timedelta(days = cpoints[i])

        # Plotting option
        if(plot):
            fig, ax = plt.subplots(figsize = FIG_SIZE)
            # Plot raw inputted data
            plt.plot(data[TIME_COL], data[feature])
            # Plot change points
            for pt in cpoints:
                plt.axvline(min_time + timedelta(days = pt), alpha = 0.3, c = 'red', dashes = (2, 2), linewidth = 2)

        # Determine phase-specific outliers and plot
        for window in new_sections:
            phase_data = data[(data[TIME_COL] > window[0]) & (data[TIME_COL] <= window[1])].copy()
            if(phase_contamination != 'auto'):
                phase_contamination = (window[1] - window[0]).days/(max_time -  min_time).days * phase_contamination
            clf = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = phase_contamination, max_features = 1.0, behaviour = 'new')
            clf.fit(phase_data[[feature]])
            info = clf.decision_function(data[[feature]])
            anomalies = clf.predict(data[[feature]])
            grouping = list(zip(phase_data.index, anomalies))
            for status_i in phase_data.index:
                if(dict(grouping)[status_i] == -1):
                    dpt = phase_data[feature][status_i]
                    if(plot):
                        ax.scatter(phase_data[TIME_COL][status_i], dpt, facecolors = 'none', edgecolors = 'r')
            grouping = list(zip(phase_data.index, anomalies, info))
            all_groups.append(grouping)
        anom_track_final = list(chain.from_iterable(all_groups))
        for tup in anom_track_final:
            if(tup[1] == -1):
                dpt = data[feature][tup[0]]
                data.at[status_i, 'anomaly'] = 'Yes'
                data.at[status_i, 'score'] = info[status_i]
                if(plot):
                    ax.scatter(data[TIME_COL][tup[0]], dpt, facecolors = 'none', edgecolors = 'r')
    plt.show()
    plt.close()
    return data, normalized_feature_data

feature_specific, normalized_all = phase_outlier_detection(df, 'SA1_SA2', 'dly_stm', 'changepoint', plot = True)

### ### ### ### ###
### ### ### ### ###
### ### ### ### ###
# fig, ax = plt.subplots(figsize = (12 * 2, 8.27 * 2))
# ax.scatter([i for i in range(len(random.sample(list(info), int(0.5 * len(info)))))], random.sample(list(info), int(0.5 * len(info))))
# fig, ax = plt.subplots(figsize = (12 * 2, 8.27 * 2))
# ax.scatter([i for i in range(len(random.sample(list(info), int(0.5 * len(info)))))], util_smooth(random.sample(list(info), int(0.5 * len(info))), 3))
# # min, max = extremas(data[feature], plot = True)
# fig, ax2 = plt.subplots(figsize = (12 * 2, 8.27 * 2))
# ax2 = ax.twinx()
# ax2.plot(data[TIME_COL], data[feature])

# FREQUENCY OF CHANGEPOINTS ACROSS GRAPHS
### ### ### ### ###
### ### ### ### ###
### ### ### ### ###
cts = []
for feature in ALL_FEATURES:
    cpts = changepoint_record_by_pad[pad][well][feature]
    cts.extend(cpts)
for feature in ALL_FEATURES:
    figure = plt.figure(figsize = (12, 8.27))
    plt.plot([i for i in range(len(data[TIME_COL]))], data[feature])
    cpts = changepoint_record_by_pad[pad][well][feature]
    plt.title(feature)
    for pt in cts:
        plt.axvline(pt, alpha = 0.3, c = 'red', dashes = (2, 2))
    for pt in cpts:
        plt.axvline(pt, alpha = 0.9, c = 'black', dashes = (2, 2), linewidth = 4)

hist, bin_edges = np.histogram(cts, bins = 100)
figure = plt.figure(figsize = (12, 8.27))
plt.plot(bin_edges[:len(bin_edges) - 1], hist)

hist = [elem * elem for elem in util_normalize(hist)]
min, max = extremas(util_smooth(hist, 3), plot = True)
### ### ### ### ###
### ### ### ### ###
### ### ### ### ###

#############################################################
#############################################################
# Statistical, NOT time-series
# Well-Independent Statistical Change Detection

# Restructuing Combination Statistics
METRICS_TO_EXCLUDE = ['window_start', 'window_end', 'percent_outliers']
delta = []
unable = []
total = 0
comparison_by_pad = {}
comparison_by_combo = {}
comparison_by_feature = {}
comparison_by_diff_metric = {}
comparison_by_deriv_metric = {}
for pad in df[PAD_COL].unique():
    df_filtered = df[df[PAD_COL] == pad]
    GROUP_COMBOS = tuple(itertools.combinations(df_filtered[GROUPBY_COL].unique(), 2))
    for combo in GROUP_COMBOS:
        for feature in ALL_FEATURES:
            stats_A, stats_B = STATS_by_group[combo[0]][feature], STATS_by_group[combo[1]][feature]
            for metric in [elem for elem in list(stats_A.columns) if elem not in METRICS_TO_EXCLUDE]:
                total += 1
                metric_A, metric_B = np.array(stats_A[metric]), np.array(stats_B[metric])
                window_A, window_B = len(metric_A), len(metric_B)
                if(window_A != window_B):
                    delta.append(window_A - window_B)
                    unable.append((pad, combo, metric, window_A, window_B, window_A - window_B))
                else:
                    metric_A, metric_B = util_normalize(metric_A), util_normalize(metric_B)
                    difference = metric_A - metric_B
                    dx = 1
                    deriv = np.diff(difference)/dx
                    comparison_by_diff_metric[metric] = difference
                    comparison_by_deriv_metric[metric] = deriv
            comparison_by_feature[feature] = {'differences': comparison_by_diff_metric.copy(), 'derivatives': comparison_by_deriv_metric.copy()}
            comparison_by_diff_metric.clear()
            comparison_by_deriv_metric.clear()
        comparison_by_combo[combo] = comparison_by_feature.copy()
        comparison_by_feature.clear()
    comparison_by_pad[pad] = comparison_by_combo.copy()
    comparison_by_combo.clear()

### SAVE DATA ###
# save_pickle('comparison_by_pad', comparison_by_pad)
comparison_by_pad = load_pickle('comparison_by_pad')

# Combo-based change detection
SMOOTH_WINDOW = 10
TWO_TAILED_MARGIN = SMOOTH_WINDOW * 2
# Although specific well and feature are specified, the list of metrics won't change
ALL_METRICS = STATS_by_group['SA1_SA2']['dly_stm'].columns
FEATURES_TO_EXCLUDE = ['runtime_hours']
i = t = 0
metric_extremas = {}
feature_extremas = {}
extremas_by_combo = {}
extremas_by_well = {}
for pad in df[PAD_COL].unique():
    df_filtered = df[df[PAD_COL] == pad]
    for well_A in df_filtered[GROUPBY_COL].unique():
        for well_B in df_filtered[GROUPBY_COL].unique():
            if(well_A == well_B):
                continue
            for feature in [elem for elem in ALL_FEATURES if elem not in FEATURES_TO_EXCLUDE]:
                for metric in [elem for elem in ALL_METRICS if elem not in METRICS_TO_EXCLUDE]:
                    t += 1
                    # try:
                    #     stats_A, stats_B = STATS_by_group[well_A][feature][metric], STATS_by_group[well_B][feature][metric]
                    # except KeyError:
                    #     continue
                    combo = comparison_by_pad[pad].keys()
                    tup = [i for i in combo if well_A in i and well_B in i][0]
                    try:
                        difference = comparison_by_pad[pad][tup][feature]['differences'][metric]
                        difference = difference[~np.isnan(difference)]
                        if(len(difference) == 0):
                            continue
                        else:
                            min_AB, max_AB = extremas(util_smooth(difference, SMOOTH_WINDOW), plot = False)
                            metric_extremas[metric] = {'minimas': dict(), 'maximas': dict()}
                            metric_extremas[metric]['minimas'].update(min_AB)
                            metric_extremas[metric]['maximas'].update(max_AB)
                        i += 1
                    except KeyError:
                        # Nothing in metric
                        continue
                feature_extremas[feature] = {'minimas': dict(), 'maximas': dict()}

                min_combined_keys = []
                min_combined_weights = []
                max_combined_keys = []
                max_combined_weights = []
                # merge all maxima/minimas and locations, metric-independent
                for metric in metric_extremas.keys():
                    min_combined_keys.extend(list(metric_extremas[metric]['minimas'].keys()))
                    min_combined_weights.extend(list(metric_extremas[metric]['minimas'].values()))
                    max_combined_keys.extend(list(metric_extremas[metric]['maximas'].keys()))
                    max_combined_weights.extend(list(metric_extremas[metric]['maximas'].values()))
                # First normalization after merge
                combined_minimas_in_all_metrics = dict(zip(min_combined_keys, util_normalize(min_combined_weights)))
                combined_maximas_in_all_metrics = dict(zip(max_combined_keys, util_normalize(max_combined_weights)))
                # Sum weights in merged, looking for repetition
                condensed_combined_minimas_in_all_metrics = {}
                condensed_combined_maximas_in_all_metrics = {}
                for key in sorted(set(combined_minimas_in_all_metrics.keys())):
                    common_keys = [k for k, v in combined_minimas_in_all_metrics.items() if k == key]
                    sum = 0
                    for ck_i in common_keys:
                        sum += combined_minimas_in_all_metrics[ck_i]
                    condensed_combined_minimas_in_all_metrics[key] = sum
                for key in sorted(set(combined_maximas_in_all_metrics.keys())):
                    common_keys = [k for k, v in combined_maximas_in_all_metrics.items() if k == key]
                    sum = 0
                    for ck_i in common_keys:
                        sum += combined_maximas_in_all_metrics[ck_i]
                    condensed_combined_maximas_in_all_metrics[key] = sum

                # Next Normalization
                norms_min_dict = dict(zip(condensed_combined_minimas_in_all_metrics.keys(), util_normalize(list(condensed_combined_minimas_in_all_metrics.values()))))
                norms_max_dict = dict(zip(condensed_combined_maximas_in_all_metrics.keys(), util_normalize(list(condensed_combined_maximas_in_all_metrics.values()))))

                feature_extremas[feature]['minimas'] = norms_min_dict
                feature_extremas[feature]['maximas'] = norms_max_dict

                extremas_by_combo[tup] = {'maximas': {}, 'minimas': {}}
                extremas_by_combo[tup]['maximas'] = feature_extremas[feature]['minimas']
                extremas_by_combo[tup]['minimas'] = feature_extremas[feature]['maximas']
    combo_iter = 0
    match_iter = 0
    well_dict_overall = {}
    for well in df_filtered[GROUPBY_COL].unique():
        maxs = {}
        mins = {}
        for combo in list(extremas_by_combo.keys()):
            if(well in combo):
                match_iter += 1
                extremas_by_well[well + '-' + str(match_iter)] = {'maximas': {}, 'minimas': {}}
                extremas_by_well[well + '-' + str(match_iter)]['maximas'] = extremas_by_combo[combo]['maximas']
                extremas_by_well[well + '-' + str(match_iter)]['minimas'] = extremas_by_combo[combo]['minimas']
        merged_keys = []
        merged_values = []
        for match_iter_num in range(1, match_iter + 1):
            curr_max = extremas_by_well[well + '-' + str(match_iter_num)]['maximas'].values()
            curr_indices = extremas_by_well[well + '-' + str(match_iter_num)]['maximas'].keys()
            merged_values.extend(curr_max)
            merged_keys.extend(curr_indices)
        well_dict = dict(zip(merged_keys, merged_values))
        for key in sorted(set(well_dict.keys())):
            common_keys = [k for k, v in well_dict.items() if k == key]
            sum = 0
            for ck_i in common_keys:
                sum += well_dict[ck_i]
            well_dict[key] = sum

        well_dict_overall[well] = {}
        well_dict_overall[well]['maximas'] = {}
        well_dict_overall[well]['maximas'] = dict(zip(well_dict.keys(), util_normalize(list(well_dict.values()))))

        merged_keys = []
        merged_values = []
        for match_iter_num in range(1, match_iter + 1):
            curr_min = extremas_by_well[well + '-' + str(match_iter_num)]['minimas'].values()
            curr_indices = extremas_by_well[well + '-' + str(match_iter_num)]['minimas'].keys()
            merged_values.extend(curr_min)
            merged_keys.extend(curr_indices)
        well_dict = dict(zip(merged_keys, merged_values))
        for key in sorted(set(well_dict.keys())):
            common_keys = [k for k, v in well_dict.items() if k == key]
            sum = 0
            for ck_i in common_keys:
                sum += well_dict[ck_i]
            well_dict[key] = sum

        well_dict_overall[well] = {}
        well_dict_overall[well]['minimas'] = {}
        well_dict_overall[well]['minimas'] = dict(zip(well_dict.keys(), util_normalize(list(well_dict.values()))))

        match_iter = 0

plt.plot(norms_min_dict.keys(), norms_min_dict.values())
plt.plot(norms_max_dict.keys(), norms_max_dict.values())

fig, ax = plt.subplots()
ax.scatter(combined_minimas_in_all_metrics.keys(), combined_minimas_in_all_metrics.values())
plt.show()

metric_extremas['mean']['minimas']
plt.plot([x for x in range(len(difference))], difference)
plt.plot([x for x in range(len(util_smooth(difference, SMOOTH_WINDOW)))], util_smooth(difference, SMOOTH_WINDOW))

# Although specific feature is specific, the time range won't matter
window_time_linkage = list(zip(STATS_by_group[well_A][feature]['window_start'], STATS_by_group[well_A][feature]['window_end']))

plot_stats(STATS_by_group['SA1_SA2']['dly_stm'], 'median_abs_deviation')
plot_stats(STATS_by_group['SA3_SA4']['dly_stm'], 'median_abs_deviation')

########################################
############# DATA STORAGE #############
# # HTML all DataFrames for storage purposes
# df_sandall.to_html('SANDALL.html')
# df_edamE.to_html('EDAM E.html')
# df_edamW.to_html('EDAM W.html')
# df_vawn.to_html('VAWN.html')
# # Pickle all DataFrames for storage purposes
# df_sandall.to_pickle('SANDALL.pkl')
# df_edamE.to_pickle('EDAM E.pkl')
# df_edamW.to_pickle('EDAM W.pkl')
# df_vawn.to_pickle('VAWN.pkl')


# Important features:
# daily steam (cumulative)
# injection + production well casing + tubing pressure (what do these mean, are they averaged (probably not cumulative))
# chloride (where's the water coming from, density comparisons) avg. or cumulative
# oil sales (what are these, what is produced) avg. or cumulative
# water sales (what are these, what is produced and recycled) avg. or cumulative
# gas sales (what are these, what is produced) avg. or cumulative
# [injection tubing temperature] why don't we have this
# production tubing temperature (why does this matter) avg. likely right?
# pump_efficiency avg. likely right?
# run-time hours avg. likely right?
# Questions:
# stm_tubing_temperature (refers to injection well?), spm_rpm (?)
# Anomaly:
# How do you find anomalies
