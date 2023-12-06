import numpy as np
import torch


def transform(ego_info, traffic_info, ego_future_path, ego_history_path):
    
    nor_x = 100
    nor_y = 100
    nor_vx = 30
    nor_vy = 30
    nor_yaw = np.pi

    ego_x = ego_info[0]
    ego_y = ego_info[1]
    ego_yaw = ego_info[4]

    x = ego_info[0]
    y = ego_info[1]
    vx = ego_info[2]
    vy = ego_info[3]
    yaw = ego_info[4]

    x_rel = x - ego_x
    y_rel = y - ego_y
    relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
    relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
    relative_vx = vx * np.cos(ego_yaw) + vy * np.sin(ego_yaw)
    relative_vy = -vx * np.sin(ego_yaw) + vy * np.cos(ego_yaw)
    relative_yaw = yaw - ego_yaw

    x_min = 972.5
    x_max = 1089.4
    y_min = 965.3
    y_max = 1034.6
    transformed_ego_info = [(ego_x - x_min)/(x_max - x_min), (ego_y - y_min)/(y_max - y_min), relative_vx / nor_vx, 
                        relative_vy / nor_vy, ego_yaw/nor_yaw]
    # transformed_ego_info = [relative_x / nor_x, relative_y / nor_y, relative_vx / nor_vx, relative_vy / nor_vy,
    #                          relative_yaw / nor_yaw]

    # return transformed_ego_info

    # 交通车
    transformed_traffic_list = []
    for traffic in traffic_info:
        x = traffic[1]
        y = traffic[2]
        vx = traffic[3]
        vy = traffic[4]
        yaw = traffic[5]

        x_rel = x - ego_x
        y_rel = y - ego_y
        relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
        relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
        relative_vx = vx * np.cos(ego_yaw) + vy * np.sin(ego_yaw)
        relative_vy = -vx * np.sin(ego_yaw) + vy * np.cos(ego_yaw)
        relative_yaw = yaw - ego_yaw

        new_traffic = [relative_x / nor_x, relative_y / nor_y, relative_vx / nor_vx, relative_vy / nor_vy,
                           relative_yaw / nor_yaw]
        
        transformed_traffic_list.append(new_traffic)

    # history path
    transformed_ego_history_path = []
    for points in ego_history_path:
        x = points[0]
        y = points[1]
        yaw = points[2]

        x_rel = x - ego_x
        y_rel = y - ego_y
        relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
        relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
        relative_yaw = yaw - ego_yaw

        new_point = [relative_x / nor_x, relative_y / nor_y, relative_yaw / nor_yaw]

        transformed_ego_history_path.append(new_point)


    # future path
    transformed_ego_future_path = []
    for point in ego_future_path:
        x = point[0]
        y = point[1]
        yaw = point[2]

        x_rel = x - ego_x
        y_rel = y - ego_y
        relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
        relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
        relative_yaw = yaw - ego_yaw

        new_point = [relative_x / nor_x, relative_y / nor_y, relative_yaw / nor_yaw]    

        transformed_ego_future_path.append(new_point)

    # ToTensor
    transformed_ego_info = torch.Tensor(transformed_ego_info)
    transformed_traffic_list = torch.Tensor(transformed_traffic_list)
    transformed_ego_future_path = torch.Tensor(transformed_ego_future_path)
    transformed_ego_history_path = torch.Tensor(transformed_ego_history_path)

    # 主车未来轨迹， 取前100个轨迹点，如果不够100，则以最后一个实际轨迹点补充到100
    if transformed_ego_future_path.shape[0] >= 100:
        transformed_ego_future_path = transformed_ego_future_path[:100]
    else:
        repeat_times = 100 - transformed_ego_future_path.shape[0]
        append_value = transformed_ego_future_path[-1][None, :].repeat(repeat_times, 1)
        transformed_ego_future_path = torch.cat((transformed_ego_future_path, append_value), axis=0)

    return transformed_ego_info, transformed_traffic_list, transformed_ego_future_path, transformed_ego_history_path


def inference(model, ego_info, traffic_info, ego_future_path, ego_history_path):
    transformed_ego_info, transformed_traffic_list, transformed_ego_future_path, transformed_ego_history_path = transform(ego_info, traffic_info, ego_future_path, ego_history_path)
    
    single_ego_veh_data = torch.unsqueeze(transformed_ego_info, 0)
    single_traffic_veh_data = torch.unsqueeze(transformed_traffic_list, 0)
    single_ego_future_track_data = torch.unsqueeze(transformed_ego_future_path, 0)
    single_ego_history_track_data = torch.unsqueeze(transformed_ego_history_path, 0)
    
    candidates_BS = 81
    if candidates_BS == 801:
        candidate_action_list = (np.arange(-5, 3.01, 0.01) + 1) / 4
        candidate_action_tensor = torch.Tensor(candidate_action_list).type_as(single_ego_veh_data)
    elif candidates_BS == 81:
        candidate_action_list = (np.arange(-5, 3.1, 0.1) + 1) / 4
        candidate_action_tensor = torch.Tensor(candidate_action_list).type_as(single_ego_veh_data)
    
    ratio_list = []
    attention_weights_list = []
    for candidate_action in candidate_action_tensor:
        candidate_action = candidate_action.unsqueeze(0)
        output = model(single_ego_veh_data, single_ego_future_track_data, single_ego_history_track_data, single_traffic_veh_data, candidate_action)
        ratio, attention_weights = output
        ratio_list.append(ratio)
        attention_weights_list.append(attention_weights)
    
    
    ratio_tensor = torch.Tensor(ratio_list)
    max_idx = torch.argmax(ratio_tensor)
    
    acc = (candidate_action_list[max_idx] * 4) - 1
    attn_weights = attention_weights_list[max_idx]
    
    return acc, attn_weights
    