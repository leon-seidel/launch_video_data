# Get live data and plots from rocket launch live streams. Correction of acceleration by Newton's law of gravitation.
#
# Arguments: --video (Video path), --start (Start time in video), --duration (Duration of video from start time), supported
# formats: 1:13:12, 3:12, 144 (h:min:s, min:s, s)) and --name (csv filename, optional).
# For livestreams just use --start live and a duration.
#
# Example 1:
# python rocket_data.py --video /file/path --start 19:53 --duration 8:24
# Example 2:
# python rocket_data.py --video /file/path --start live --duration 8:45 --name test.csv --type SpaceX --title IFT-5

import os
import re
import cv2
import time
import argparse
import pytesseract
import numpy as np
import pandas as pd
from sys import platform
from matplotlib import pyplot as plt

if platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def get_rocket_data(arguments):
    ##################################################################################################################
    # Performance ####################################################################################################
    every_n = 15                            # Only analyse every nth frame
    # Plot settings ##################################################################################################
    upper_limit_velo_plot = 28000           # Upper limit of velocity plot
    upper_limit_alti_plot = 200             # Upper limit of altitude plot
    lower_limit_acc_plot = -45              # Lower limit of acceleration plot
    upper_limit_acc_plot = 45               # Upper limit of acceleration plot
    # Outlier prevention #############################################################################################
    lower_limit_acc = -70                   # Highest negative acceleration in m/s^2
    upper_limit_acc = 70                    # Highest positive acceleration in m/s^2
    lower_limit_v_vert = -12                # Highest negative vertical velocity in km/s
    upper_limit_v_vert = 12                 # Highest positive vertical velocity in km/s
    tresh_v_vert = 0.5                      # Vertical velocity is multiplied with this value before comparison to v
    mean_of_last = 15                       # Mean value of last n acceleration values
    # Telemetry data sources #########################################################################################
    # contains [y_start, y_end, x_start, x_end] of the bounding box ##################################################
    f9_stage1 = [0.88889, 0.930556, 0.053125, 0.206250]         # Position of telemetry data in 720p video feed (Falcon 9, stage 1)
    f9_stage2 = [0.88889, 0.930556, 0.793750, 0.942969]      # Position of telemetry data in 720p video feed (Falcon 9, stage 2)
    rocketlab = [0.04861, 0.076389, 0.762500, 0.878125]        # Position of telemetry data in 720p video feed (Rocket Lab Electron)
    jwst = [0.752778, 0.951389, 0.132813, 0.193750]             # Position of telemetry data in 720p video feed (JWST stream Arianespace)
    # labpadre = [0, 30, 1140, 1205]          # Position of clock in livestream (just for livestream testing)
    # astra = [654, 677, 1045, 1180]          # position of telemetry data in 720p video feed (NSF - Astra)
    starship = [0.842593, 0.907407, 0.800521, 0.84375]      # Position of telemetry data in 1080p video feed (Starship/ Falcon 9, stage 2))
    super_heavy = [0.842593, 0.907407, 0.177083, 0.234375]      # Position of telemetry data in 1080p video feed (Super Heavy/ Falcon 9, stage 1))
    # Constants ######################################################################################################
    gc = 6.6723e-11                         # Gravitational constant (m^3/kgs^2)
    m_earth = 5.972e24                      # Earth mass (kg)
    r_earth = 6.371e6                       # Earth radius (m)
    ##################################################################################################################

    video_name = arguments.video
    video_start_time, video_end_time, video_duration, is_live = get_video_times(arguments)

    video_title = arguments.title
    video_type = arguments.type

    t, v, h, a, v_vert, a_mean = [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]

    if video_type == "SpaceX_old":
        pos_stage_rel = [f9_stage1, f9_stage2]
        velo_unit = "kph"
    elif video_type == "RocketLab":
        pos_stage_rel = [rocketlab]
        velo_unit = "kph"
    elif video_type == "arianespace":
        pos_stage_rel = [jwst]
        velo_unit = "kph"
    # elif video_type == "LabPadre":
    #     pos_stage_rel = [labpadre]
    #     velo_unit = "kph"
    # elif video_type == "NASASpaceflight":
    #     pos_stage_rel = [astra]
    #     velo_unit = "ms"
    elif video_type == "SpaceX":
        pos_stage_rel = [super_heavy, starship]
        velo_unit = "kph"
    else:
        pos_stage_rel = None
        velo_unit = "kph"
        print("Video type " + video_type + " not supported.")
        quit()

    number_of_stages = len(pos_stage_rel)  # Number of rocket stages with data

    fig, ax, sc = start_plots(number_of_stages, video_title, upper_limit_velo_plot, upper_limit_alti_plot,
                              upper_limit_acc_plot, lower_limit_acc_plot, video_duration, t, v, h, a_mean)

    cap = cv2.VideoCapture(video_name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    pos_stage = pos_stage_rel
    for n, stage_rel in enumerate(pos_stage_rel):
        pos_stage[n] = [round(height*stage_rel[0]), round(height*stage_rel[1]), round(width*stage_rel[2]), round(width*stage_rel[3])]

    if is_live is False:
        cap.set(cv2.CAP_PROP_POS_MSEC, video_start_time * 1000)
        true_video_end_time = video_end_time * 1000
    else:
        true_video_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) + video_duration * 1000

    start_time = time.time()
    frame_number = 0
    frame_time = 0  # Time between video start and T0
    p = 0
    
    while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= true_video_end_time:

        p += 1
        frame_number += 1

        frame_time += 1 / fps

        if p != every_n:
            cap.grab()
            continue
        else:
            ret, frame = cap.read()
            p = 0

        if frame is None:
            print("\nVideo ended here.")
            break

        t_frame = round(frame_time, 3)
        print()

        for stage in range(1, number_of_stages + 1):
            # v_frame in km/h, h_frame in km
            v_frame, h_frame = get_text_from_frame(video_type, frame, pos_stage, stage)

            # Change velocity unit if it is not kp/h
            if velo_unit == "ms" and v_frame is not None:
                v_frame = v_frame * 3.6     # If v_frame is in m/s

            # a_read_frame in m/s^2: veloity change rate
            a_read_frame = calculate_acc(t, v, t_frame, v_frame, stage)
            # v_vert_frame in km/s: vertical velocity
            v_vert_frame = calculate_v_vert(t, h, t_frame, h_frame, stage)
            # v_hori_frame in km/s: horizontal velocity
            v_hori_frame = calculate_v_hori(v_frame, v_vert_frame)
            # a_corr_frame in m/s^2: gravitational acceleration by Newton's law of universal gravitation
            # Using a = MG/r^2 - v^2/r (derived by difference between mMG/r^2 and mv^2/r)
            a_corr_frame = calculate_a_corr(h_frame, v_hori_frame, gc, m_earth, r_earth)

            # a_frame in m/s^2: sum of read acceleration and gravitational acceleration
            if a_read_frame is not None and a_corr_frame is not None:
                a_frame = a_read_frame + a_corr_frame
            elif a_read_frame is not None and a_corr_frame is None:
                a_frame = a_read_frame
            else:
                a_frame = None

            # Outlier detection: Check if a_frame and v_vert_frame are within their predefined boundaries and if
            # v_vert_frame is lower than v_frame. The multiplier thres_v_vert is used to avoid skipping values where
            # v_vert_frame and v are close togheter.
            # For stage 2: Check if v_frame and h_frame are higher than for stage 1.
            if (a_frame is not None and v_vert_frame is not None and v_frame is not None and
                    lower_limit_acc <= a_frame <= upper_limit_acc and lower_limit_v_vert <=
                    v_vert_frame <= upper_limit_v_vert and (v_vert_frame * 3600 * tresh_v_vert) <= v_frame):
                if stage == 2 and v_frame is not None:
                    try:
                        n = 1
                        while v[0][-n] is None or h[0][-n] is None:
                            n += 1
                        if v_frame < v[0][-n] or h_frame < h[0][-n]:
                            continue
                    except IndexError:
                        t[stage - 1].append(None)
                        v[stage - 1].append(None)
                        h[stage - 1].append(None)
                        a[stage - 1].append(None)
                        v_vert[stage - 1].append(None)
                        a_mean[stage - 1].append(None)
                        continue
                    except TypeError:
                        t[stage - 1].append(None)
                        v[stage - 1].append(None)
                        h[stage - 1].append(None)
                        a[stage - 1].append(None)
                        v_vert[stage - 1].append(None)
                        a_mean[stage - 1].append(None)
                        continue

                t[stage - 1].append(t_frame)
                v[stage - 1].append(v_frame)
                h[stage - 1].append(h_frame)
                a[stage - 1].append(a_frame)
                v_vert[stage - 1].append(v_vert_frame)

                a_frame_mean = calculate_a_mean(a, stage, mean_of_last)

                a_mean[stage - 1].append(a_frame_mean)

                print("Stage " + str(stage) + ": t= " + str(t_frame) + " s, v= " + str(v_frame) + " kph, h= " +
                      str(h_frame) + " km, a= " + str(a_frame_mean) + " m/s^2")
            else:
                t[stage - 1].append(None)
                v[stage - 1].append(None)
                h[stage - 1].append(None)
                a[stage - 1].append(None)
                v_vert[stage - 1].append(None)
                a_mean[stage - 1].append(None)

        time_passed = time.time() - start_time
        average_fps = frame_number / time_passed

        print("Average fps: " + str(round(average_fps, 2)) + ", total time: " + str(round(time_passed, 2)) + " s")

        update_plots(number_of_stages, t, v, h, a_mean, fig, sc)

    cap.release()
    print("\nFinished!")
    save_as_csv(t, v, h, a_mean, number_of_stages, video_title, arguments.csv)

    plt.waitforbuttonpress()


def get_video_times(arguments):
    video_start_time, video_end_time, video_duration, is_live = 0, 0, 0, False

    if ":" in arguments.duration:
        duration_list = arguments.duration.split(":")

        if len(duration_list) == 2:
            video_duration = float(duration_list[0]) * 60 + float(duration_list[1])
        if len(duration_list) == 3:
            video_duration = float(duration_list[0]) * 3600 + float(duration_list[1]) * 60 + float(duration_list[2])
    else:
        video_duration = float(arguments.duration)

    if arguments.start == "live":
        is_live = True
        video_end_time = video_duration
        return video_start_time, video_end_time, video_duration, is_live
    else:
        if ":" in arguments.start:
            start_list = arguments.start.split(":")

            if len(start_list) == 2:
                video_start_time = float(start_list[0]) * 60 + float(start_list[1])
            if len(start_list) == 3:
                video_start_time = float(start_list[0]) * 3600 + float(start_list[1]) * 60 + float(start_list[2])
        else:
            video_start_time = float(arguments.start)

        video_end_time = video_start_time + video_duration
        return video_start_time, video_end_time, video_duration, is_live


def start_plots(number_of_stages, video_title, upper_limit_velo_plot, upper_limit_alti_plot, upper_limit_acc_plot,
                lower_limit_acc_plot, video_duration, t, v, h, a_mean):
    fig, ax, sc = [], [], [[], [], []]

    # Velocity plot
    plt.ion()
    fig_velo, ax_velo = plt.subplots()
    fig.append(fig_velo)
    ax.append(ax_velo)

    for stage in range(1, number_of_stages + 1):
        sc[0].append(ax[0].scatter(t[stage - 1], v[stage - 1]))
        if stage == 2:
            plt.legend(["Stage 1", "Stage 2"])

    plt.title(video_title + ": Time vs. velocity")
    plt.xlim(0, video_duration)
    plt.ylim(0, upper_limit_velo_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Velocity in kph")
    plt.grid()
    plt.draw()

    # Altitude plot
    plt.ion()
    fig_alti, ax_alti = plt.subplots()
    fig.append(fig_alti)
    ax.append(ax_alti)

    for stage in range(1, number_of_stages + 1):
        sc[1].append(ax[1].scatter(t[stage - 1], h[stage - 1]))
        if stage == 2:
            plt.legend(["Stage 1", "Stage 2"])

    plt.title(video_title + ": Time vs. altitude")
    plt.xlim(0, video_duration)
    plt.ylim(0, upper_limit_alti_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Altitude in km")
    plt.grid()
    plt.draw()

    # Acceleration plot
    plt.ion()
    fig_acc, ax_acc = plt.subplots()
    fig.append(fig_acc)
    ax.append(ax_acc)

    for stage in range(1, number_of_stages + 1):
        sc[2].append(ax[2].scatter(t[stage - 1], a_mean[stage - 1]))
        if stage == 2:
            plt.legend(["Stage 1", "Stage 2"])

    plt.title(video_title + ": Time vs. acceleration")
    plt.xlim(0, video_duration)
    plt.ylim(lower_limit_acc_plot, upper_limit_acc_plot)
    plt.xlabel("Time in s")
    plt.ylabel("Acceleration in m/s^2")
    plt.grid()
    plt.draw()

    return fig, ax, sc


def get_text_from_frame(video_type, frame, pos_stage, stage):
    cropped = frame[pos_stage[stage - 1][0]:pos_stage[stage - 1][1], pos_stage[stage - 1][2]:pos_stage[stage - 1][3]]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    text_list = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", text)

    if len(text_list) == 2:

        try:
            v_frame = float(text_list[0])
        except ValueError:
            v_frame = None

        try:
            h_frame = float(text_list[1])
        except ValueError:
            h_frame = None

    elif video_type == "arianespace" and len(text_list) == 4:
        try:
            v_frame = round((float(text_list[2]) + (float(text_list[3]) / 100)) * 3600, 1)
        except ValueError:
            v_frame = None

        try:
            h_frame = float(text_list[0])
        except ValueError:
            h_frame = None

    else:
        v_frame = None
        h_frame = None

    return v_frame, h_frame


def calculate_acc(t, v, t_frame, v_frame, stage):
    try:
        m = 0
        while True:
            m += 1
            if not [x for x in (v[stage - 1][-m], t[stage - 1][-m], v_frame, t_frame) if x is None]:
                a_frame = ((v_frame - v[stage - 1][-m]) / 3.6) / (t_frame - t[stage - 1][-m])
                return a_frame
            elif [x for x in (v_frame, t_frame) if x is None]:
                a_frame = None
                return a_frame
    except IndexError:
        a_frame = 0
        return a_frame


def calculate_v_vert(t, h, t_frame, h_frame, stage):
    try:
        m = 0
        while True:
            m += 1
            if not [x for x in (h[stage - 1][-m], t[stage - 1][-m], h_frame, t_frame) if x is None]:
                v_vert_frame = (h_frame - h[stage - 1][-m]) / (t_frame - t[stage - 1][-m])
                return v_vert_frame
            elif [x for x in (h_frame, t_frame) if x is None]:
                v_vert_frame = None
                return v_vert_frame
    except IndexError:
        v_vert_frame = 0
        return v_vert_frame


def calculate_v_hori(v_frame, v_vert_frame):
    if v_frame is None or v_vert_frame is None:
        v_hori_frame = None
    elif v_frame < 0 or v_vert_frame < 0:
        v_hori_frame = None
    elif v_vert_frame >= v_frame / 3600:
        v_hori_frame = 0
    else:
        v_frame_kms = v_frame / 3600        # Total velocity in km/s
        v_hori_frame = np.sqrt(np.square(v_frame_kms) - np.square(v_vert_frame))

    return v_hori_frame     # Horizontal velocity in km/s


def calculate_a_corr(h_frame, v_hori_frame, gc, m_earth, r_earth):
    # Using a = MG/r^2 - v^2/r (derived by difference between mMG/r^2 and mv^2/r)
    if h_frame is None or v_hori_frame is None:
        a_corr_frame = None
    else:
        h_frame_m = h_frame * 1000                  # Altitude in m
        v_hori_frame_ms = v_hori_frame * 1000       # Horzontal velocity in m/s

        a_corr_frame = (m_earth * gc) / np.square((h_frame_m + r_earth)) - np.square(v_hori_frame_ms) / \
                       (h_frame_m + r_earth)

    return a_corr_frame                             # Correction of acceleration in m/s^2


def calculate_a_mean(a, stage, mean_of_last):
    try:
        n = 0
        m = 0
        n_last = []
        while n < mean_of_last:
            m += 1
            if a[stage - 1][-m] is not None:
                n_last.append(a[stage - 1][-m])
                n += 1
        a_frame_mean = round(np.mean(n_last), 3)
    except IndexError:
        a_frame_mean = None
    except TypeError:
        a_frame_mean = None

    return a_frame_mean


def update_plots(number_of_stages, t, v, h, a_mean, fig, sc):
    for stage in range(1, number_of_stages + 1):
        sc[0][stage - 1].set_offsets(np.c_[t[stage - 1], v[stage - 1]])
    fig[0].canvas.draw_idle()
    plt.pause(0.001)

    for stage in range(1, number_of_stages + 1):
        sc[1][stage - 1].set_offsets(np.c_[t[stage - 1], h[stage - 1]])
    fig[1].canvas.draw_idle()
    plt.pause(0.001)

    for stage in range(1, number_of_stages + 1):
        sc[2][stage - 1].set_offsets(np.c_[t[stage - 1], a_mean[stage - 1]])
    fig[2].canvas.draw_idle()
    plt.pause(0.001)


def save_as_csv(t, v, h, a_mean, number_of_stages, video_title, filename):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = 'mission_data'

    if filename is None:
        csv_filename = "".join(x for x in video_title if x.isalnum()) + ".csv"
    else:
        csv_filename = filename

    csv_file = os.path.join(file_dir, csv_folder, csv_filename)

    if number_of_stages == 1:
        column_names = ["t", "v", "h", "a"]
        df_list = list(zip(t[0], v[0], h[0], a_mean[0]))
    elif number_of_stages == 2:
        column_names = ["t", "v1", "h1", "a1", "v2", "h2", "a2"]
        df_list = list(zip(t[0], v[0], h[0], a_mean[0], v[1], h[1], a_mean[1]))
    else:
        df_list, column_names = None, None
        print("Writing a csv of more than two stages is not supported.")
        quit()

    df = pd.DataFrame(df_list, columns=column_names)
    df.to_csv(csv_file, index=False)

    print("\nSaved data to " + csv_folder + "/" + csv_filename + "!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read and plot data from SpaceX F9 and Rocket Lab Electron starts')

    parser.add_argument('--video', nargs='?', type=str, help='Video name')
    parser.add_argument('--type', nargs='?', type=str, help='Type of video from "SpaceX", "SpaceX_old", "RocketLab", "arianespace"')
    parser.add_argument('--start', nargs='?', type=str, default="0", help='Video start time, formats: 1:13:12, 3:12, 144, live')
    parser.add_argument('--duration', nargs='?', type=str, help='Video duration, formats: 1:13:12, 3:12, 144')
    parser.add_argument('--csv', nargs='?', type=str, help='Name for csv file, for example ABLaunch25_07_21.csv')
    parser.add_argument('--title', nargs='?', type=str, default="Launch", help='Title in plots')

    args = parser.parse_args()

    if args.video is None:
        print("Pleade add a video name with --video /path/to/video")
        quit()
    if args.duration is None:
        print("Pleade add a video duration, supported formats: 1:13:12, 3:12, 144")
        quit()

    get_rocket_data(args)
