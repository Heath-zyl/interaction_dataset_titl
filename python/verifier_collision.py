from shapely.geometry import LineString, Point, Polygon
import numpy as np


def verifier_collision(ego_x, ego_y, ego_yaw, traffic_veh_list):
    collision_flag = False

    length, width = 5, 2

    front_left, front_right, rear_left, rear_right = \
        calculate_vertices(ego_x, ego_y, ego_yaw, length, width)
    ego_polygon = Polygon([front_left, front_right, rear_right, rear_left])

    for obstacle in traffic_veh_list:
        traffic_x = obstacle[1]
        traffic_y = obstacle[2]
        traffic_yaw = obstacle[5]
        outrange = abs(ego_x - traffic_x) > 15 or abs(ego_y - traffic_y) > 15
        if outrange:
            continue
        traffic_front_left, traffic_front_right, traffic_rear_left, traffic_rear_right = \
            calculate_vertices(traffic_x, traffic_y, traffic_yaw, length, width)
        traffic_polygon = \
            Polygon([traffic_front_left, traffic_front_right, traffic_rear_right, traffic_rear_left])
        if ego_polygon.intersects(traffic_polygon):
            collision_flag = True
            break
    return collision_flag


def calculate_vertices(x, y, yaw, length, width):
    # 计算车辆的四个顶点坐标
    # 车辆坐标系下，前方为x轴正方向，右侧为y轴正方向

    # 计算车辆的四个顶点在车辆坐标系下的坐标
    front_left = [-length / 2, -width / 2]
    front_right = [-length / 2, width / 2]
    rear_left = [length / 2, -width / 2]
    rear_right = [length / 2, width / 2]

    # 根据航向角yaw旋转车辆的四个顶点
    front_left_rotated = [
        front_left[0] * np.cos(yaw) - front_left[1] * np.sin(yaw),
        front_left[0] * np.sin(yaw) + front_left[1] * np.cos(yaw)
    ]

    front_right_rotated = [
        front_right[0] * np.cos(yaw) - front_right[1] * np.sin(yaw),
        front_right[0] * np.sin(yaw) + front_right[1] * np.cos(yaw)
    ]

    rear_left_rotated = [
        rear_left[0] * np.cos(yaw) - rear_left[1] * np.sin(yaw),
        rear_left[0] * np.sin(yaw) + rear_left[1] * np.cos(yaw)
    ]

    rear_right_rotated = [
        rear_right[0] * np.cos(yaw) - rear_right[1] * np.sin(yaw),
        rear_right[0] * np.sin(yaw) + rear_right[1] * np.cos(yaw)
    ]

    # 计算顶点在全局坐标系下的坐标
    front_left_global = (front_left_rotated[0] + x, front_left_rotated[1] + y)
    front_right_global = (front_right_rotated[0] + x, front_right_rotated[1] + y)
    rear_left_global = (rear_left_rotated[0] + x, rear_left_rotated[1] + y)
    rear_right_global = (rear_right_rotated[0] + x, rear_right_rotated[1] + y)

    return front_left_global, front_right_global, rear_left_global, rear_right_global
