#!/usr/bin/env python

try:
    import lanelet2

    use_lanelet2_lib = True
except ImportError:
    import warnings

    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils


def update_plot():
    global fig, timestamp, title_text, track_dictionary, patches_dict, text_dict, axes, pedestrian_dictionary, displacement_error
    
    # timestamp: 当前帧数
    # title_text: 标题行
    # track_dictionary: 全部轨迹字典，不变
    # patches_dict: 车辆贴图
    # text_dict: 车辆贴图上的标号
    # axes: 坐标，不变
    
    # update text and tracks based on current timestamp
    assert (timestamp <= timestamp_max), "timestamp=%i" % timestamp
    assert (timestamp >= timestamp_min), "timestamp=%i" % timestamp
    assert (timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    percentage = (float(timestamp) / timestamp_max) * 100
    
    if displacement_error is None:
        title_text.set_text("\nts = {} / {} ({:.2f}%)".format(timestamp, timestamp_max, percentage))
    elif timestamp in displacement_error:
        sort_key = list(displacement_error.keys())
        min_key, max_key = sort_key[0], sort_key[-1]
        
        if timestamp == max_key:
            title_text.set_text("\nts = {} / {} ({:.2f}%); FDE@{}ms={:.3f} ".format(timestamp, timestamp_max, percentage, timestamp-min_key, displacement_error[timestamp]))
        else:
            title_text.set_text("\nts = {} / {} ({:.2f}%); displacement_error@{}ms={:.3f} ".format(timestamp, timestamp_max, percentage, timestamp-min_key, displacement_error[timestamp]))
    else:
        sort_key = list(displacement_error.keys())
        min_key, max_key = sort_key[0], sort_key[-1]
        title_text.set_text("\nts = {} / {} ({:.2f}%); FDE@{}ms={:.3f}".format(timestamp, timestamp_max, percentage, (max_key-min_key), displacement_error[max_key]))
    
    tracks_vis.update_objects_plot(timestamp, patches_dict, text_dict, axes,
                                   track_dict=track_dictionary, pedest_dict=pedestrian_dictionary)
    fig.canvas.draw()


def start_playback():
    global timestamp, timestamp_min, timestamp_max, playback_stopped
    playback_stopped = False
    plt.ion()
    while timestamp < timestamp_max and not playback_stopped:
        timestamp += dataset_types.DELTA_TIMESTAMP_MS
        start_time = time.time()
        update_plot()
        end_time = time.time()
        diff_time = end_time - start_time
        plt.pause(max(0.001, dataset_types.DELTA_TIMESTAMP_MS / 1000. - diff_time))
    plt.ioff()


class FrameControlButton(object):
    def __init__(self, position, label):
        self.ax = plt.axes(position)
        self.label = label
        self.button = Button(self.ax, label)
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        global timestamp, timestamp_min, timestamp_max, playback_stopped

        if self.label == "play":
            if not playback_stopped:
                return
            else:
                start_playback()
                return
        playback_stopped = True
        if self.label == "<<":
            timestamp -= 10 * dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == "<":
            timestamp -= dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">":
            timestamp += dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">>":
            timestamp += 10 * dataset_types.DELTA_TIMESTAMP_MS
        timestamp = min(timestamp, timestamp_max)
        timestamp = max(timestamp, timestamp_min)
        update_plot()


if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    # parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
    #                                                     "files)", nargs="?")
    # parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
    #                                                     "files)", default="")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="both",
                        nargs="?")
    
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--track_id", type=int, help='track id to present')
    # parser.add_argument("--duration", type=float, default=5.0, help='duration for present')
    
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    args = parser.parse_args()

    # print(args) 
    # # Namespace(lat_origin=0.0, load_mode='both', lon_origin=0.0, scenario_name='DR_USA_Intersection_MA', start_timestamp=None, track_file_number=0)

    scenario_name = 'DR_USA_Intersection_MA'

    if scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # print(scenario_name, args.load_mode) # DR_USA_Intersection_MA both

    # check folders and files
    error_string = ""

    # root directory is one above main_visualize_data.py file
    # i.e. the root directory of this project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    maps_dir = os.path.join(root_dir, "maps")

    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, scenario_name + lanelet_map_ending)

    scenario_dir = os.path.join(tracks_dir, scenario_name)

    track_file_name = os.path.join(
        scenario_dir,
        "vehicle_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
    )
    pedestrian_file_name = os.path.join(
        scenario_dir,
        "pedestrian_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
    )

    if not os.path.isdir(tracks_dir):
        error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(maps_dir):
        error_string += "Did not find map file directory \"" + maps_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(lanelet_map_file):
        error_string += "Did not find lanelet map file \"" + lanelet_map_file + "\"\n"
    if not os.path.isfile(track_file_name):
        error_string += "Did not find track file \"" + track_file_name + "\"\n"
    if not os.path.isfile(pedestrian_file_name):
        flag_ped = 0
    else:
        flag_ped = 1
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)
    
    print('*' * 120)
    print(f'tracks_dir: {tracks_dir}')
    print(f'tracks_dir: {maps_dir}')
    print(f'scenario_dir: {scenario_dir}')
    print(f'lanelet_map_file: {lanelet_map_file}')
    print(f'track_file_name: {track_file_name}')
    print(f'pedestrian_file_name: {pedestrian_file_name}')
    print('*' * 120)

    # create a figure
    fig, axes = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Interaction Dataset Visualization")

    # load and draw the lanelet2 map, either with or without the lanelet2 library
    lat_origin = args.lat_origin  # origin is necessary to correctly project the lat lon values of the map to the local
    lon_origin = args.lon_origin  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    # print(lat_origin) # 0.0
    # print(lon_origin) # 0.0
    
    print("Loading map...")
    if use_lanelet2_lib:
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
    else:
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)

    # load the tracks
    print("Loading tracks...")
    track_dictionary = None
    pedestrian_dictionary = None
    if args.load_mode == 'both': # this branch
        track_dictionary = dataset_reader.read_tracks(track_file_name)
        if flag_ped:
            pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)
    elif args.load_mode == 'vehicle':
        track_dictionary = dataset_reader.read_tracks(track_file_name)
    elif args.load_mode == 'pedestrian':
        pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)

    # assert args.track_id is None or isinstance(args.track_id, int), 'track_id must be a digit.'
    # assert args.track_id is None or args.track_id in track_dictionary.keys(), 'wrong track_id.'

    # print(track_dictionary.keys())
    # print(track_dictionary[1])

    ######### START #########
    assert args.track_id is None and args.start_timestamp is None
    import numpy as np
    import ast
    import torch
    from collections import OrderedDict
    from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
    from tqdm import tqdm
    
    # 读取主车路径文件: ego_path_dict_file
    ego_path_dict_file = 'utils_folder/ego_path_dict.npy'
    ego_path_dict = np.load(ego_path_dict_file, allow_pickle=True)
    ego_path_dict = str(ego_path_dict)
    ego_path_dict = ast.literal_eval(ego_path_dict)
    ego_path_dict = dict(ego_path_dict)

    # 读取交通车信息文件: traffic_dict_file
    traffic_dict_file = 'utils_folder/track_dict.npy'
    traffic_dict = np.load(traffic_dict_file, allow_pickle=True)
    traffic_dict = str(traffic_dict)
    traffic_dict = ast.literal_eval(traffic_dict)
    traffic_dict = dict(traffic_dict)

    # 创建 model
    d_model = 16
    nhead = 4
    num_layers = 1
    # model_path = 'model_ckpt/epoch_999_无负样本.pth'
    model_path = 'model_ckpt/epoch_119_负样本500.pth'
    print(model_path)
    model = CarTrackTransformerEncoder(num_layers=num_layers, nhead=nhead, d_model=d_model)
    weights = torch.load(model_path, map_location='cpu')
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()
    
    # 计算 ade 和 fde
        
    ade_list = []
    fde_list = []

    from main_calulate_prediction import process
    for track_id, info in tqdm(track_dictionary.items()):
        
        if track_id % 10 != 1:
            continue
        
        start_frame, end_frame = info.time_stamp_ms_first // 100, info.time_stamp_ms_last // 100
        # 10frame ~ 1s
        
        for frame in range(start_frame, end_frame, 10):
            # print(f'  ==> stating time_stamp: {frame*100}')
            
            if (frame + 51) * 100 > info.time_stamp_ms_last:
                break
            
            prediction_track_dict = process(ego_id=track_id, init_frame_id=frame, predicting_frames=31, ego_path_dict=ego_path_dict, traffic_dict=traffic_dict, model=model)
            if prediction_track_dict is None:
                break
            
            assert len(prediction_track_dict.keys()) == 31, len(prediction_track_dict.keys())
            
            # get ade
            for time_stamp_ms, pred_motion_states in prediction_track_dict.items():
                gt_x, gt_y = info.motion_states[time_stamp_ms].x, info.motion_states[time_stamp_ms].y
                pred_x, pred_y = pred_motion_states.x, pred_motion_states.y
                ade = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
                ade_list.append(ade)
            
            # get fde
            last_timestamp_in_this_window = sorted(list(prediction_track_dict.keys()))[-1]
            gt_x, gt_y = prediction_track_dict[last_timestamp_in_this_window].x, prediction_track_dict[last_timestamp_in_this_window].y
            pred_x, pred_y = info.motion_states[last_timestamp_in_this_window].x, info.motion_states[last_timestamp_in_this_window].y
            fde = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
            fde_list.append(fde)
        
    final_ade = sum(ade_list) / len(ade_list)
    final_fde = sum(fde_list) / len(fde_list)
    
    print(f'ADE: {final_ade}')
    print(f'FDE: {final_fde}')
    
    import sys
    sys.exit()
        
        # print()
        # print('========')
    
        # import sys
        # sys.exit()
    
    # assert args.track_id is None
    
    # Calculate prediction track
    # displacement_error = {}
    # if args.track_id is not None:
    #     track_id = int(args.track_id)
    #     print('caculate prediction track...')
    #     from main_calulate_prediction import process
    #     prediction_track_dict = process(ego_id=track_id, init_frame_id=args.start_timestamp//100, predicting_frames=int(args.duration*10))
        
    #     from copy import deepcopy
    #     car_ego_copy = deepcopy(track_dictionary[track_id])
        
    #     track_dictionary[str(track_id)+'_ego'] = track_dictionary.pop(track_id)

    #     car_ego_prediction = car_ego_copy

    #     for key, value in prediction_track_dict.items():
    #         car_ego_prediction.motion_states[key] = value

    #     car_ego_prediction.track_id = str(car_ego_copy.track_id) + 'auto'

    #     track_dictionary[car_ego_prediction.track_id] = car_ego_prediction
        
        
    #     for i, timestamp in enumerate(range(args.start_timestamp, args.start_timestamp+int(args.duration*1000)+100, 100)):
    #         if timestamp not in track_dictionary[str(track_id)+'_ego' ].motion_states:
    #             break
    #         a = track_dictionary[str(track_id)+'_ego' ].motion_states[timestamp]
    #         b = track_dictionary[car_ego_copy.track_id].motion_states[timestamp]

    #         abs_dis = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
    #         displacement_error[timestamp] = abs_dis

    #     # print(displacement_error)

    # else:
    #     args.duration = None

    ######### END #########


    timestamp_min = 1e9
    timestamp_max = 0
    
    if track_dictionary is not None:
        for key, track in dict_utils.get_item_iterator(track_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last)
    else:
        for key, track in dict_utils.get_item_iterator(pedestrian_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last)
            

    if args.start_timestamp is None:
        args.start_timestamp = timestamp_min
        
    if args.duration:
        timestamp_max = min(timestamp_max, int(args.start_timestamp+args.duration*1000))

    button_pp = FrameControlButton([0.2, 0.05, 0.05, 0.05], '<<')
    button_p = FrameControlButton([0.27, 0.05, 0.05, 0.05], '<')
    button_f = FrameControlButton([0.4, 0.05, 0.05, 0.05], '>')
    button_ff = FrameControlButton([0.47, 0.05, 0.05, 0.05], '>>')

    button_play = FrameControlButton([0.6, 0.05, 0.1, 0.05], 'play')
    button_pause = FrameControlButton([0.71, 0.05, 0.1, 0.05], 'pause')

    # storage for track visualization
    patches_dict = dict()
    text_dict = dict()

    # visualize tracks
    print("Plotting...")
    timestamp = args.start_timestamp
    title_text = fig.suptitle("")
    playback_stopped = True
    update_plot()
    plt.show()
