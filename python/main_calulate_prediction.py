import numpy as np
import ast
from scipy.interpolate import interp1d
from utils.dataset_types import MotionState
from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
import torch
from collections import OrderedDict, defaultdict
from model_inference import inference
from tqdm import tqdm


def process(ego_id, init_frame_id, ego_path_dict_file='utils_folder/ego_path_dict.npy', traffic_dict_file='utils_folder/track_dict.npy', predicting_frames=50):
    # 读取主车路径文件: ego_path_dict_file
    ego_path_dict = np.load(ego_path_dict_file, allow_pickle=True)
    ego_path_dict = str(ego_path_dict)
    ego_path_dict = ast.literal_eval(ego_path_dict)
    ego_path_dict = dict(ego_path_dict)

    # 读取交通车信息文件: traffic_dict_file
    traffic_dict = np.load(traffic_dict_file, allow_pickle=True)
    traffic_dict = str(traffic_dict)
    traffic_dict = ast.literal_eval(traffic_dict)
    traffic_dict = dict(traffic_dict)

    # 创建 model
    d_model = 16
    nhead = 4
    num_layers = 1
    # model_path = 'model_ckpt/epoch_119_负样本500.pth'
    # model_path = 'model_ckpt/epoch_473_负样本2000.pth'
    # model_path = 'model_ckpt/epoch_999_无负样本.pth'
    # model_path = 'model_ckpt/epoch_51_负样本20.pth'
    model_path = 'model_ckpt/epoch_999_负样本1000.pth'
    
    model = CarTrackTransformerEncoder(num_layers=num_layers, nhead=nhead, d_model=d_model)
    weights = torch.load(model_path, map_location='cpu')
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()

    # 获得主车初始状态
    init_ego_x, init_ego_y, init_ego_yaw, init_ego_vx, init_ego_vy, init_S = get_inital_state(ego_id, init_frame_id, ego_path_dict)    
    # init_ve = get_ve(init_ego_vx, init_ego_vy, init_ego_yaw)
    init_ve = init_ego_vx * np.cos(init_ego_yaw) + init_ego_vy * np.sin(init_ego_yaw)

    # 获得初始历史轨迹
    init_ego_history_path = initial_ego_history(init_ego_x, init_ego_y, init_ego_yaw)
    cur_ego_history_path = init_ego_history_path

    # 循环初始状态
    cur_frame_id = init_frame_id
    cur_S, cur_ego_x, cur_ego_y = init_S, init_ego_x, init_ego_y
    cur_vx, cur_vy = init_ego_vx, init_ego_vy
    cur_yaw = init_ego_yaw
    cur_ve = init_ve
    
    # 循环N*10次(N秒)
    prediction_track_dict = {}
    trackID2_to_frameAttn_dict = defaultdict(list)
    for i in tqdm(range(predicting_frames)):
        
        if f'{ego_id}-{cur_frame_id}' not in traffic_dict:
            # print(f'there is no {ego_id}-{cur_frame_id}.')
            break
        
        traffic_info = traffic_dict[f'{ego_id}-{cur_frame_id}']
        # print(len(traffic_info), type(traffic_info))
        # print(traffic_info)
        
        # 获得主车未来轨迹
        ego_future_path = get_future_path(ego_id, ego_path_dict, cur_S)
        # print(len(ego_future_path))
        
        # 获得主车预测加速度
        # ego_info = (cur_ego_x, cur_ego_y, cur_vx, cur_vy, cur_yaw)
        # acc = inference(model, ego_info=ego_info, traffic_info=traffic_info, ego_future_path=ego_future_path)
        ego_info = (cur_ego_x, cur_ego_y, cur_vx, cur_vy, cur_yaw)
        
        acc, attn_weights = inference(model, ego_info=ego_info, traffic_info=traffic_info, ego_future_path=ego_future_path, ego_history_path=cur_ego_history_path)
        
        # attn_weights: [1, 1, N, N]
        
        cls_weight = attn_weights[0][0, 0, 4:]
        
        # Normalize to 0-1
        # mmax, mmin = cls_weight.max(), cls_weight.min()
        # cls_weight = (cls_weight - mmin) / (mmax - mmin)
        
        # Weight
        cls_weight = cls_weight / (cls_weight.sum())
        
        # gamma trans + weight_trans
        # cls_weight = cls_weight ** 8
        # cls_weight = cls_weight / (cls_weight.sum())
        
        for traffic, attn in zip(traffic_info, cls_weight):
            traffic_id = traffic[0]
            trackID2_to_frameAttn_dict[traffic_id].append(f'{cur_frame_id*100}-{"%.4f"%attn}')
        
        # 根据预测得到下一步路径
        next_S, next_ve = get_s_ve(cur_S, cur_ve, acc)

        try:
            # 根据下一步路径得到下一步路径的状态
            next_ego_x, next_ego_y, next_vx, next_vy, next_yaw = get_state(ego_id, next_S, old_x=cur_ego_x, old_y=cur_ego_y, ego_path_dict=ego_path_dict)
        except Exception as e:
            break
            
        # next_ve = get_ve(next_vx, next_vy, next_yaw)
        next_ego_history_path = update_ego_history(next_ego_x, next_ego_y, next_yaw, cur_ego_history_path)
        
        cur_frame_id += 1
        cur_S = next_S
        cur_ve = next_ve
        cur_ego_x, cur_ego_y = next_ego_x, next_ego_y
        cur_vx, cur_vy = next_vx, next_vy
        cur_yaw = next_yaw
        cur_ego_history_path = next_ego_history_path
                
        new_motion_state = MotionState(time_stamp_ms=cur_frame_id*1000)
        new_motion_state.x = cur_ego_x
        new_motion_state.y = cur_ego_y
        new_motion_state.vx = cur_vx
        new_motion_state.vy = cur_vx
        new_motion_state.psi_rad = cur_yaw
        prediction_track_dict[cur_frame_id*100] = new_motion_state
    
    return prediction_track_dict, trackID2_to_frameAttn_dict
    
    
def get_state(ego_id, S, old_x, old_y, ego_path_dict):
    Ts = 0.1
    ego_path = ego_path_dict[str(ego_id)]
    
    x = [point[1] for point in ego_path]
    y = [point[2] for point in ego_path]
    yaw = [point[3] for point in ego_path]
    distances = [point[-1] for point in ego_path]

    # 创建插值函数
    f_x = interp1d(distances, x, kind='linear')
    f_y = interp1d(distances, y, kind='linear')
    f_yaw = interp1d(distances, yaw, kind='linear')

    # 定义新的路程值（等路程间隔）
    new_distances = S

    # 使用插值函数计算对应的 x、y 值和航向角
    new_x = f_x(new_distances)
    new_y = f_y(new_distances)
    new_yaw = f_yaw(new_distances)
    new_vx = (new_x - old_x) / Ts
    new_vy = (new_y - old_y) / Ts
    return new_x, new_y, new_vx, new_vy, new_yaw


def get_s_ve(S0, Ve, acc):
    # S0是车辆当前的路程，Ve是车辆当前的速度，acc是车辆当前的加速度
    Ts = 0.1  # 时间步长
    S = S0 + Ve * Ts + 1/2 * acc * Ts * Ts
    new_Ve = Ve + acc * Ts
    
    return S, new_Ve

 
# def get_ve(vx, vy, ego_yaw):
#     return vx * np.cos(ego_yaw) + vy * np.sin(ego_yaw)
 
 
def get_inital_state(ego_id, frame_id, ego_path_dict):
    # 根据主车id和frame_id获得初始状态
    ego_path = ego_path_dict[str(ego_id)]    
    
    for i, frame in enumerate(ego_path):
        if frame[0] == frame_id:
            ego_x, ego_y, ego_yaw, ego_vx, ego_vy, S = frame[1:]
            break
        
    return ego_x, ego_y, ego_yaw, ego_vx, ego_vy, S


def get_future_path(ego_id, ego_path_dict, S):
    ego_path = ego_path_dict[str(ego_id)]

    x = [point[1] for point in ego_path]
    y = [point[2] for point in ego_path]
    yaw = [point[3] for point in ego_path]
    distances = [point[-1] for point in ego_path]

    # 创建插值函数
    f_x = interp1d(distances, x, kind='linear')
    f_y = interp1d(distances, y, kind='linear')
    f_yaw = interp1d(distances, yaw, kind='linear')

    # 定义新的路程值（等路程间隔）
    new_distances = np.arange(S, distances[-1], 0.5)  # 此处步长为1.0，可以根据需要调整

    # 使用插值函数计算对应的 x、y 值和航向角
    new_x = f_x(new_distances)
    new_y = f_y(new_distances)
    new_headings = f_yaw(new_distances)
    new_ego_future_path = list(zip(new_x, new_y, new_headings))

    return new_ego_future_path


def initial_ego_history(x, y, yaw):
    # oldest_point = (x, y, yaw)
    # ego_history_path = [oldest_point] * 10
    # return ego_history_path


    ##########
    ##########
    ##########

    ego_history_path = input('please input ego history path: ')
    # print(ego_history_path)
    # ego_history_path = ego_history_path.strip()[1:-1]
    # print(ego_history_path.split(','))
    
    ego_history_path = ''.join([char for char in ego_history_path if char not in [' ', '(', ')', '[', ']']]).split(',')
    
    ego_history_path = [(float(ego_history_path[i]), float(ego_history_path[i+1]), float(ego_history_path[i+2])) for i in range(0, len(ego_history_path), 3)]
    
    return ego_history_path
    

def update_ego_history(x, y, yaw, ego_history_path):
    ego_history_path.pop(0)
    ego_history_path.append((x, y, yaw))
    return ego_history_path
