########################################
########## DEPENDENCY IMPORTS ##########
import pandas as pd
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import sys
import numpy as np
import os
import subprocess
import glob
from scipy.stats import mode
from datetime import datetime
import time

########################################
########### HYPER PARAMETERS ###########
sys.path.append("/Users/Ray/Documents/Python/9 - Oil and Gas/Sandall/")
sys.path.append('/Users/Ray/Documents/Python/9 - Oil and Gas/Sandall/')
PATH_SANDALL = 'Data/Sandall/sandall.csv'
PATH_EDAM_EAST = 'Data/Edam/edam_east.csv'
PATH_EDAM_WEST = 'Data/Edam/edam_west.csv'
PATH_VAWN = 'Data/Vawn/vawn_pvr_daily_2020_03_09.csv'

RESOLUTION_BINS = 40
FIG_SIZE = (220, 7)

########################################
############ LOCAL FUNCTIONS ###########
def generate_video(imgs, name):
    folder = "/Users/Ray/Documents/Python/9 - Oil and Gas/"
    for i in range(len(imgs)):
        axes = plt.gca()
        axes.set_title(name)
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        plt.hist(imgs[i], bins = RESOLUTION_BINS, color = 'skyblue')
        plt.savefig(folder + "temp/file%02d.png" % i)

    os.chdir('/Users/Ray/Documents/Python/9 - Oil and Gas/temp')
    subprocess.call([
        'ffmpeg', '-framerate', '60', '-i', 'file%02d.png', '-r', '120', '-pix_fmt', 'yuv420p', name + '_ANIMATION.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    plt.close()

def generate_scratch_animation(df, foi):
    images = []
    traversed_dates = []
    for w in df['pair_name'].unique():
        filtered_df = df[df['pair_name'] == w]
        xmin = min(filtered_df[foi])
        xmax = max(filtered_df[foi])
        ymin = 0
        ymax = mode(filtered_df[foi])[1]
        for d in df[df['pair_name'] == w]['production_date'].unique():
            traversed_dates.append(d)
            histplot = df[(df['pair_name'] == w) & (df['production_date'].isin(traversed_dates))][foi]
            # mode_loc = mode(histplot)[0][0]
            # mode_count = mode(histplot)[1][0]
            # if(low data in distribution):
            #     continue
            images.append(histplot)
            # pp.savefig(histplot.get_figure())
        print('ANIMATION >> Current Image List Dimension:' + str(len(images)))
        generate_video(images[50:250], w + "-" + foi)
        images.clear()
        traversed_dates.clear()

# Generates time-dependent animation of specified feature after grouping is applied
def generate_depn_animation(df, groupby, time_feature, foi, labels, fig_size = (12.5, 9), resolution = 'high'):
    images = []
    traversed_dates = []
    START_TIME = datetime.now()

    # For every well...
    for w in df[groupby].unique():
        # GENERATE DIRECTORIES
        # Create encapsulating folder `Animations` ifnexists
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/Animations'):
            os.makedirs(cwd + '/Animations')
        # Determine path of final video save
        if(resolution == 'high'):
            save_dir = cwd + "/Animations/HRES/"
        elif(resolution == 'low'):
            save_dir = cwd + "/Animations/LRES/"
        elif(resolution == 'auto'):
            save_dir = cwd + "/Animations/AUTO/"
        else:
            print('ANIMATION >> **FATAL ERROR**: Resolution Argument (' + resolution + ") Incorrectly Specified")
            return -1
        # Make sub-folder for specific well
        save_dir += w + '/'
        # Create folder of final video save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # If FILE already exists, assume file is in acceptable format
        # Skip vidoe generation due to computational expensiveness, don't override
        if(os.path.exists(save_dir + foi + '.mp4')):
            print("ANIMATION >> File Already Exists : " + save_dir + foi + ".mp4")
            continue

        filtered_df = df[df[groupby] == w]
        xmin = min(filtered_df[foi])
        xmax = max(filtered_df[foi])
        ymin = 0
        ymax = mode(filtered_df[foi])[1]

        print('ANIMATION >> ' + w + ': Determining Frames...')
        for d in filtered_df[time_feature].unique():
            traversed_dates.append(d)
            hist = df[(df[groupby] == w) & (df[time_feature].isin(traversed_dates))][foi]
            images.append([hist])

        print('ANIMATION >> Image Vector Dimension: ' + str(len(images)))
        def update_hist(num, images):
            plt.cla()
            plt.title(w + ": " + foi + "\n" + traversed_dates[num], fontsize = 20)
            plt.xlabel(labels[0], fontsize = 14)
            plt.ylabel(labels[1], fontsize = 14)

            if(resolution == 'high'):
                # Binning dependent on globally define variable
                plt.hist(images[num], bins = RESOLUTION_BINS)
            elif(resolution == 'low'):
                # Default value for bins
                plt.hist(images[num], bins = 10)
            elif(resolution == 'auto'):
                # Uses the maximum of the Sturges and Freedman-Diaconis bin choice
                plt.hist(images[num], bins = 'auto')
            else:
                print('ANIMATION >> **FATAL ERROR**: Resolution Argument (' + resolution + ") Incorrectly Specified")
                return -1

        print('ANIMATION >> Processing Animation...')
        fig = plt.figure()
        fig.set_size_inches(*fig_size, True)
        hist = plt.hist(images[0], bins = RESOLUTION_BINS)
        anima = animation.FuncAnimation(fig, update_hist, len(images), fargs = (images, ), interval = 4)
        print('ANIMATION >> Completed Video Processing.')

        print('ANIMATION >> Saving Video...')

        # Ultimately save the video w/ all specified directories
        # OPTIONAL (takes very long w/ usage, not huge benefit): writer = animation.FFMpegFileWriter(bitrate = -1)
        anima.save(save_dir + foi + '.mp4', dpi = 200)
        print('ANIMATION >> ' + w + ': Video Saved.\n')

        images.clear()
        traversed_dates.clear()
        process_logger(record = 'FOI: ' + foi + "\t" + "Well: " + w, START_TIME = START_TIME)

        plt.close()

# Returns DataFrame with applied column replacements, operation NOT inplace
def repl_df_cols(df_original, dict_repl, save = True, name = None):
    df_field_cleaned = df_original.copy()
    df_field_cleaned = df_field_cleaned.rename(columns = dict_repl)
    if(save):
        df_field_cleaned.to_csv(name + '.csv')
    return df_field_cleaned

# For each unique pair name/well, make histogram of each of the features
def write_freq_matrix(df, groupby, mpl_PDF, features_filter):
    print('FREQ. MATRIX >> Processing Matrix...')
    for w in df[groupby].unique():
        histplot = df[df[groupby] == w].hist(layout = (1, len(features_filter)), figsize = FIG_SIZE, bins = RESOLUTION_BINS)
        plt.suptitle(w, fontsize = 41)
        mpl_PDF.savefig(histplot[0][0].get_figure())
    print('FREQ. MATRIX >> Confirming Matrix Process...')

# For each unique pair name/well, produce time-dependent plots of given selected feature
def write_ts_matrix(df, groupby, time_feature, mpl_PDF, features_filter):
    print('T.S. MATRIX >> Processing Matrix...')
    for w in df[groupby].unique():
        tsplot = df[df[groupby] == w].plot(title = w, x = time_feature, subplots = True, layout = (1, len(features_filter)), figsize = FIG_SIZE)
        plt.suptitle(w, fontsize = 41)
        mpl_PDF.savefig(tsplot[0][0].get_figure())
    print('T.S. MATRIX >> Confirming Matrix Process...')

# Major process logger
def process_logger(record, START_TIME, name = 'script_logger.txt'):
    if not os.path.exists(name):
        text_file = open(name, "w")
    else:
        text_file = open(name, "a+")

    elapsed_time = datetime.now() - START_TIME
    last_time = START_TIME + elapsed_time
    delta_time = datetime.now() - last_time

    text_file.writelines('Elapsed: ' + str(elapsed_time) + "\t Delta: " + str(delta_time) + ":: " + record + '\n')
    text_file.close()


########################################
########### DATA PROCESSING ############
df_sandall = pd.read_csv(PATH_SANDALL)
# df_sandall_cleaned = repl_df_cols(df_sandall, {'job_id': 'job'}, name = 'sandall_cleaned')
df_edamE = pd.read_csv(PATH_EDAM_EAST)
df_edamW = pd.read_csv(PATH_EDAM_WEST)
df_vawn = pd.read_csv(PATH_VAWN)
df_ALL = [df_sandall, df_edamE, df_edamW, df_vawn]

# Updating the elements in a list inplace will intentionally mutate the original
[df.drop('Unnamed: 0', 1, inplace = True) for df in df_ALL]

# Features to incorporate into charts, validation PDF
features_to_check = [
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

# Set field of interest
df = df_sandall

# Setup features and initialize PDF
pvr = df[['production_date', 'pair_name'] + features_to_check]

########## ########### ###########

features_to_animate = pvr.columns
features_to_animate = [e for e in features_to_animate if e not in ('production_date', 'pair_name')]
for res_option in ['high', 'low', 'auto']:
    for col in features_to_animate:
        feature_of_interest = col
        # generate_scratch_animation(pvr, feature_of_interest)
        # generate_depn_animation(pvr, feature_of_interest)
        generate_depn_animation(pvr, groupby = 'pair_name', time_feature = 'production_date',
                                     foi = feature_of_interest, labels = ["Unit", "Frequency"], resolution = res_option)

# Frequency and Time Series Generation of all variables
pp = PdfPages('Sandall_Validation.pdf')
write_freq_matrix(pvr, 'pair_name', pp, features_to_check)
write_ts_matrix(pvr, 'pair_name', 'production_date', pp, features_to_check)
pp.close()




#
