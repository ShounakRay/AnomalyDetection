# @Author: Shounak Ray <Ray>
# @Date:   27-Oct-2020 11:10:35:357  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: Anomaly_Detection_PKG.py
# @Last modified by:   Ray
# @Last modified time: 11-Feb-2021 15:02:87:871  GMT-0700
# @License: [Private IP]


from datetime import datetime, timedelta
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from stringcase import snakecase

# import eif

###### UTILITY FUNCTIONS ######
# convert from normal text to snake case for all columns in df


def util_snakify_cols(df):
    df.columns = [snakecase(col).replace('__', '_') for col in df.columns]
    return df
# Return cleaned DataFrame


def reset_df_index(df):
    return df.reset_index().drop('index', 1)
# Normalize list


def util_normalize(list):
    list = (list - np.min(list)) / (np.max(list) - np.min(list))
    return list

###### MAIN FUNCTION ######
# "Online phase detection"


def step_outlier_detection(data, well, feature, ALL_FEATURES=['None'], method='Offline Outlier', mode='overall',
                           gamma='scale', nu='0.5', model_name='rbf', diff_thresh=256, N_EST=100, contamination='0.1',
                           TIME_COL='production_date', GROUPBY_COL='pair_name', plot=False, seed=42, n_jobs=-1):
    # Snakify columns and feature name
    data = util_snakify_cols(data)
    feature, TIME_COL, GROUPBY_COL = snakecase(feature).replace('__', '_'), snakecase(TIME_COL).replace('__', '_'), snakecase(GROUPBY_COL).replace('__', '_')

    # Data-type verification and variable settings
    FIG_SIZE = (12, 8.27)
    data = pd.DataFrame(data)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data[TIME_COL] = data[TIME_COL].apply(lambda x: x.date())
    well = str(well)
    feature = str(feature)
    if(method not in ['Online Novelty', 'Offline Outlier', 'Offline DBSCAN']):
        raise ValueError('XXX `method` wrong input. XXX')
    if(mode not in ['changepoint', 'overall']):
        raise ValueError('XXX `mode` wrong input. XXX')
    elif(mode == 'changepoint'):
        net_contamination = None
        if(contamination != 'auto'):
            phase_contamination = float(contamination)
            if(phase_contamination < 0.0 or phase_contamination > 1.0):
                raise ValueError('XXX `phase_contamination` is outside 0-1 range. XXX')
        else:
            phase_contamination = 'auto'
    elif(mode == 'overall'):
        phase_contamination = None
        if(contamination != 'auto'):
            net_contamination = float(contamination)
            if(net_contamination < 0.0 or net_contamination > 1.0):
                raise ValueError('XXX `net_contamination` is outside 0-1 range. XXX')
        else:
            net_contamination = 'auto'
    if(model_name not in ['rbf', 'l1', 'l2']):
        if(method == 'Offline Outlier'):
            raise ValueError('XXX `model_name` not properly inputted for Offline Outlier Method. XXX')
        else:
            if(model_name not in ['linear', 'poly', 'sigmoid', 'precomputed']):
                raise ValueError('XXX `model_name` not properly inputted for Online Novelty Method. XXX')
    diff_thresh = int(diff_thresh)
    N_EST = int(N_EST)
    TIME_COL = str(TIME_COL)
    seed = int(seed)
    if(gamma not in ['scale', 'auto']):
        if(isinstance(s, str)):
            raise ValueError("XXX Gamma not correctly inputted. XXX")
        else:
            gamma = float(gamma)
    nu = float(nu)
    if('None' in ALL_FEATURES):
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
    else:
        pass

    # High-level data re-structuring and spec filtering
    data = reset_df_index(data[data[GROUPBY_COL] == well]).sort_values(by=TIME_COL)
    data.drop([GROUPBY_COL], 1, inplace=True)
    data.to_html('DATA_FILE_ANOMALY.html')
    # Filters for normalized DataFrame
    print([TIME_COL] + ALL_FEATURES)
    print(data)
    normalized_feature_data = data.copy()[[TIME_COL] + ALL_FEATURES]
    for ft in ALL_FEATURES:
        normalized_feature_data[ft] = util_normalize(normalized_feature_data[ft])

    # Filter main DataFrame
    # !!! NOTE THESE COLUMN NAMES MUST BE SNAKE_CASE COMPATIBLE
    data = data[[TIME_COL, feature]]
    data.columns = [TIME_COL, 'selection']
    data['anomaly'] = 'No'

    def outlier_detection_iforest(data, contamination, n_estimators=N_EST, max_samples='auto', max_features=1.0,
                                  random_state=seed, n_jobs=-1):
        clf = IsolationForest(n_estimators=N_EST, max_samples='auto', contamination=contamination,
                              max_features=1.0, random_state=seed, n_jobs=-1)
        clf.fit(data[['selection']])
        info = clf.decision_function(data[['selection']])
        anomalies = clf.predict(data[['selection']])
        return info, anomalies

    def novetly_detection_OCSVM(data, model_name, gamma=gamma, nu=nu, TIME_COL=TIME_COL):
        # Data Processing
        min_time = min(data[TIME_COL])
        numerical_dates = data[TIME_COL].apply(lambda x: (x - min_time).days).tolist()
        feature_values = data['selection'].tolist()
        X_train = np.array(list(zip(numerical_dates, feature_values)))
        # SVM Classifier Configurations
        clf = OneClassSVM(kernel=model_name, gamma=gamma, nu=nu)
        clf.fit(X_train)
        anomalies = clf.predict(X_train)
        info = clf.decision_function(X_train)
        return info, anomalies

    def outlier_detection_DBSCAN(data, eps=0.9, min_samples=5, leaf_size=100, n_jobs=-1):
        clf = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=n_jobs)
        clf.fit(np.array(data['selection']).reshape(-1, 1))
        outlier_index = np.where(clf.labels_ == -1)[0]
        anomalies = [1] * len(data)
        for i in outlier_index:
            anomalies[i] = -1
        return anomalies, anomalies

    # Analyze whole dataset for outlier detection
    if(mode == 'overall'):
        # Full-based Outlier Detection
        if(method == 'Offline Outlier'):
            info, anomalies = outlier_detection_iforest(data, n_estimators=N_EST, max_samples='auto',
                                                        contamination=net_contamination, max_features=1.0,
                                                        random_state=seed, n_jobs=-1)
        elif(method == 'Online Novelty'):
            info, anomalies = novetly_detection_OCSVM(data, model_name, gamma=gamma, nu=nu)
        elif(method == 'Offline DBSCAN'):
            info, anomalies = outlier_detection_DBSCAN(data, eps=0.5, min_samples=5, leaf_size=30, n_jobs=-1)

        if(plot):
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            plt.plot(data[TIME_COL], data['selection'])
        anoms_internal = []
        score_internal = []
        for status_i in range(len(anomalies)):
            if(anomalies[status_i] == -1):
                dpt = data['selection'][status_i]
                anoms_internal.append('Yes')
                score_internal.append(info[status_i])
                if(plot):
                    ax.scatter(data[TIME_COL][status_i], dpt, facecolors='none', edgecolors='r')
            else:
                anoms_internal.append('No')
                score_internal.append(max(info))
        data['anomaly'] = anoms_internal.copy()
        data['scores'] = score_internal.copy()
        if(plot):
            plt.title(well + ", " + feature + ", " + mode)
            plt.show()
    # Analyze dataset for outlier detection in segments
    else:
        # Phase-based Outlier Detection
        min_time = min(data[TIME_COL])
        max_time = max(data[TIME_COL])
        new_sections = []
        all_groups = []
        now_time = min_time

        # Find change points and determine windows
        cpoints = rpt.Pelt(model=model_name).fit(np.array(data['selection'])).predict(pen=3)
        cpoints = [cpoints[i] for i in range(len(cpoints) - 1) if cpoints[i + 1] >= cpoints[i] + diff_thresh]
        if((max_time - min_time).days + 1 not in cpoints):
            cpoints.insert(len(cpoints), (max_time - min_time).days + 1)
        for i in range(len(cpoints)):
            new_sections.append((now_time, min_time + timedelta(days=cpoints[i])))
            now_time = min_time + timedelta(days=cpoints[i])

        # Plotting option
        if(plot):
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            # Plot raw inputted data
            plt.plot(data[TIME_COL], data['selection'])
            # Plot change points
            for pt in cpoints:
                plt.axvline(min_time + timedelta(days=pt), alpha=0.3, c='red', dashes=(2, 2), linewidth=2)

        # Determine phase-specific outliers and plot
        for window in new_sections:
            if(window == new_sections[0]):
                phase_data = data[(data[TIME_COL] >= window[0]) & (data[TIME_COL] <= window[1])].copy()
            else:
                phase_data = data[(data[TIME_COL] > window[0]) & (data[TIME_COL] <= window[1])].copy()
            # if(phase_contamination != 'auto'):
            #     phase_contamination = (window[1] - window[0]).days/(max_time -  min_time).days * phase_contamination

            if(method == 'Offline Outlier'):
                info, anomalies = outlier_detection_iforest(data, n_estimators=N_EST, max_samples='auto',
                                                            contamination=phase_contamination, max_features=1.0,
                                                            random_state=seed, n_jobs=-1)
            elif(method == 'Online Novelty'):
                info, anomalies = novetly_detection_OCSVM(data, model_name, gamma=gamma, nu=nu)
            elif(method == 'Offline DBSCAN'):
                info, anomalies = outlier_detection_DBSCAN(data, eps=0.5, min_samples=5, leaf_size=30, n_jobs=-1)
            grouping = list(zip(phase_data.index, anomalies))
            for status_i in phase_data.index:
                if(dict(grouping)[status_i] == -1):
                    dpt = phase_data['selection'][status_i]
                    if(plot):
                        ax.scatter(phase_data[TIME_COL][status_i], dpt, facecolors='none', edgecolors='r')
            grouping = list(zip(phase_data.index, anomalies, info))
            all_groups.append(grouping)
        anom_track_final = list(chain.from_iterable(all_groups))
        anoms_internal = []
        score_internal = []
        for tup in anom_track_final:
            if(tup[1] == -1):
                dpt = data['selection'][tup[0]]
                anoms_internal.append('Yes')
                score_internal.append(info[tup[0]])
                if(plot):
                    ax.scatter(data[TIME_COL][tup[0]], dpt, facecolors='none', edgecolors='r')
            else:
                anoms_internal.append('No')
                score_internal.append(max(info))
        data['anomaly'] = anoms_internal.copy()
        data['scores'] = score_internal.copy()
        if(plot):
            plt.title(well + ", " + feature + ", " + mode + ", " + str(diff_thresh))
            plt.show()
    plt.close()

    # Map anomalies to normalized DataFrame
    normalized_feature_data['anomaly_map'] = data['anomaly']
    normalized_feature_data['score_map'] = data['scores']
    f_maximas = []
    for row in range(len(normalized_feature_data)):
        if(normalized_feature_data.iloc[row]['anomaly_map'] == 'Yes'):
            f_maximas.append(np.float64(np.max(list(normalized_feature_data.iloc[row][ALL_FEATURES]))))
        else:
            f_maximas.append(np.float64(0.0))
    normalized_feature_data['frame_maximas'] = f_maximas
    if(mode == 'changepoint'):
        ns_lena = len(new_sections)
        all_states = pd.DataFrame(new_sections)
        all_states = all_states.append(pd.DataFrame({0: all_states[1][len(all_states) - 1],
                                                     1: all_states[1][len(all_states) - 1]},
                                                    index=[len(all_states)]))
        all_states[1] = max(data['selection'])
        all_states[2] = 1.0
        all_states.columns = ['changepoint', 'regular_y', 'norm_y']
    else:
        ns_lena = 'N/A'
        all_states = pd.DataFrame([{0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}],
                                  columns=['changepoint', 'regular_y', 'norm_y'])
        all_states['changepoint'],
        all_states['regular_y'],
        all_states['norm_y'] = [data[TIME_COL].iloc[0], data[TIME_COL].iloc[len(data) - 1]],
        [max(data['selection']), max(data['selection'])],
        [1.0, 1.0]

    normalized_feature_data.rename(columns={feature: 'selection'}, inplace=True)

    return reset_df_index(data), reset_df_index(normalized_feature_data), ns_lena, reset_df_index(all_states)

# Complete anomaly detection with repetitions


def anomaly_detection(data, well, feature, ALL_FEATURES=['None'], method=['Offline Outlier'],
                      mode=['overall', 'changepoint'], gamma='scale', nu='0.5', model_name='rbf', diff_thresh=100,
                      N_EST=100, contamination=['0.1'], TIME_COL='production_date', GROUPBY_COL='pair_name',
                      plot=False, seed=42, n_jobs=-1, iteration=1):
    iteration = int(iteration)
    ft = data[(data[GROUPBY_COL] == well)].sort_values(by=TIME_COL)
    all_dates = list(data[TIME_COL])
    detect_track = []
    # Re-assign `iteration` if method/mode/contamination lengths are greater than argument
    if(iteration < len(method) or iteration < len(mode) or iteration < len(contamination)):
        print('XXX `iteration` is lower than model specifications. Argument re-assigned. XXX')
        iteration = max(len(method), len(mode), len(contamination))
    # Re-assign `iteration` if method/mode lengths are lower than argument
    if(iteration > len(method) or iteration > len(mode) or iteration > len(contamination)):
        print('XXX `iteration` is greater than model specifications. Argument re-assigned. XXX')
        iteration = max(len(method), len(mode), len(contamination))

    # Duplicate entered mode to match size of method specifications
    if(len(mode) == 1 and (len(method) > 1 or len(contamination) > 1)):
        mode = [mode[0] for i in range(max(len(method), len(contamination)))]
    # Duplicate entered method to match size of mode specifications
    elif(len(method) == 1 and (len(mode) > 1 and len(contamination) > 1)):
        method = [method[0] for i in range(max(len(mode), len(contamination)))]
    elif(len(contamination) == 1 and (len(mode) > 1 and len(method) > 1)):
        contamination = [contamination[0] for i in range(max(len(mode), len(method)))]

    # If method/mode sizes are not 1, but not equal, auto-complete the shorter spec list
    elif(len(set([len(mode), len(method), len(contamination)])) > 1):
        max_length = max([len(mode), len(method), len(contamination)])
        relation = {'mode': mode, 'method': method, 'contamination': contamination}
        for i in relation.values():
            if(len(relation[i]) < max_length):
                last = relation[i][-1]
                relation[i] = [relation[i].append(last) for i in range(max_length - len(relation[i]))]
        # if(len(method) > len(mode)):
        #     last = str(mode[-1])
        #     for i in range(len(method) - len(mode)):
        #         mode.append(last)
        # elif(len(method) < len(mode)):
        #     last = str(method[-1])
        #     for i in range(len(mode) - len(method)):
        #         method.append(last)
        mode, method, contamination = relation['mode'], relation['method'], relation['contamination']

    for iter in range(iteration):
        ft, total, new_sections, windows = step_outlier_detection(data, well, feature, ALL_FEATURES, method[0],
                                                                  mode[0], gamma, nu, model_name, diff_thresh, N_EST,
                                                                  contamination[0], TIME_COL, GROUPBY_COL, plot, seed,
                                                                  n_jobs)

        data = reset_df_index(data[data[GROUPBY_COL] == well]).sort_values(by=TIME_COL)
        for row in range(len(ft)):
            if(ft.iloc[row]['anomaly'] == 'Yes'):
                data.drop(row, 0, inplace=True)
        if(plot):
            fig, ax = plt.subplots(figsize=(12, 9))
            plt.plot(ft[ft['anomaly'] == 'No'][TIME_COL], ft[ft['anomaly'] == 'No']['selection'])
            plt.title('Iteration ' + str(iter + 1))
            ax.set_xlim(ft[TIME_COL][0], ft[TIME_COL][-1:])
            ax.set_ylim(min(ft['selection']), 500)
            plt.show()

        ft_dates = list(ft[ft['anomaly'] == 'Yes'][TIME_COL].copy())

        if(iter == 0):
            base = ft.copy()
            base_norm = total.copy()
            base_len = new_sections
            base_windows = windows.copy()
            for i in range(len(base)):
                if(base.iloc[i]['anomaly'] == 'Yes'):
                    detect_track.append(str(iter + 1))
                else:
                    detect_track.append(str(-1))
            base['detection_iter'] = detect_track.copy()
            base['anomaly'] = ['Yes' if int(i) >= 1 else 'No' for i in detect_track.copy()]
        else:
            new_detect_track = []
            for i in range(len(base)):
                date_i = base.iloc[i][TIME_COL]
                if(base.iloc[i]['anomaly'] == 'Yes'):
                    # Datum was detected as an outlier in any previous iteration
                    # > Re-assign last assigned iteration tag to new/current tracker
                    new_detect_track.append(str(detect_track[i]))
                else:
                    # Datum wasn't detected as an outlier in any previous iteration
                    if(date_i in ft_dates):
                        # But datum is detected as an outlier in current iteration
                        new_detect_track.append(str(iter + 1))
                    else:
                        # Datum wasn't detected in current or previous iteration, it's not an outlier
                        new_detect_track.append(str(-1))
            base['detection_iter'] = new_detect_track.copy()
            base['anomaly'] = ['Yes' if int(i) >= 1 else 'No' for i in new_detect_track.copy()]
            detect_track = new_detect_track.copy()

    base_norm['detection_iter'] = detect_track.copy()
    base_norm['anomaly_map'] = ['Yes' if int(i) >= 1 else 'No' for i in detect_track].copy()

    cpoints_final = list(base_windows['changepoint'])
    cpoint_status = []
    cpoint_y = []
    cpoint_max = np.float64(base_windows['regular_y'][0])
    for i in range(len(base)):
        base_row_date = base[TIME_COL][i]
        if(base_row_date in cpoints_final):
            # Date is a changepoint
            cpoint_status.append('Yes')
            cpoint_y.append(cpoint_max)
        else:
            # Date is not a changepoint
            cpoint_status.append('No')
            cpoint_y.append(np.float64(0.0))
    base['changepoint_status'] = cpoint_status.copy()
    base['regular_y'] = [cpoint_max if i == 'Yes' else np.float64(0.0) for i in cpoint_status]
    base_norm['changepoint_status'] = cpoint_status.copy()
    base_norm['regular_y'] = [np.float64(1.0) if i == 'Yes' else np.float64(0.0) for i in cpoint_status]

    information = pd.DataFrame([{}])
    information['well_name'] = well
    information['feature_name'] = feature
    information['pct_anomalies'] = str(round((len(base[base['anomaly'] == 'Yes']) / (len(base))) * 100.0, 3)) + '%'
    information['states'] = base_len

    plt.close()

    return base, base_norm, information, base_windows

# # # # FUNCTION CALL
# __FOLDER__ = r'/Users/Ray/Documents/Python/9 - Oil and Gas/Husky'
# PATH_SANDALL = __FOLDER__ + r'/Data/Sandall/sandall.csv'
# PATH_EDAM_EAST = __FOLDER__ + r'/Data/Edam/edam_east.csv'
# data = pd.read_csv(PATH_SANDALL)
# data = reset_df_index(data.dropna(inplace = False))
# # Settings
# well = 'SA1_SA2'
# feat = 'dly_stm'
# mtds = ['Offline Outlier']
# mds = ['overall', 'overall']
# cnts = ['0.2', '0.2']
#
# ft, total, info, windows = anomaly_detection(data, well, feat, ALL_FEATURES = ['Default'], method = mtds, mode = mds,
#                                              gamma = 'scale', nu = 0.3, model_name = 'rbf', N_EST = 100,
#                                              diff_thresh = 100, contamination = cnts, plot = True, n_jobs = -1,
#                                              iteration = 1)
#
# # #
# # # # OUT-OF-FUNCTION MODEL VERIFICATION
# # Print original plot
# fig, ax = plt.subplots(figsize = (15, 10))
# ts = reset_df_index(data[(data['pair_name'] == well)].sort_values(by = 'production_date'))
# plt.plot(ts['production_date'], ts[feat])
# ax.set_xlim(ts['production_date'][0], ts['production_date'][-1:])
# ax.set_ylim(0, max(data[feat]))
# plt.show()
# plt.close()
#
# # Print final plot
# fig, ax = plt.subplots(figsize = (15, 10))
# plt.plot(ft[ft['anomaly'] == 'No']['production_date'], ft[ft['anomaly'] == 'No']['selection'])
# ax.scatter(ft[ft['anomaly'] == 'No']['production_date'], ft[ft['anomaly'] == 'No']['selection'], s = 7)
# ax.set_xlim(ft['production_date'][0], ft['production_date'][-1:])
# ax.set_ylim(0, max(data[feat]))
# fig, ax = plt.subplots(figsize = (15, 10))
# plt.plot(ft[ft['anomaly'] == 'No']['production_date'], ft[ft['anomaly'] == 'No']['selection'])
# ax.scatter(ft[ft['anomaly'] == 'No']['production_date'], ft[ft['anomaly'] == 'No']['selection'], s = 7)
# ax.scatter(ft[ft['anomaly'] == 'Yes']['production_date'], ft[ft['anomaly'] == 'Yes']['selection'],
#            facecolors = 'none', edgecolors = 'r', s = 10)
# ax.set_xlim(ft['production_date'][0], ft['production_date'][-1:])
# ax.set_ylim(0, max(data[feat]))
# plt.show()
# plt.close()

#
