# Python Scripts for the INTERACTION dataset


`1. 特定帧下,特定主车的碰撞检测`

    python ./python/main_visualize_data.py 0 \
      --start_timestamp 263300 # 帧数
      --track_id 145 # 主车id
      --duration 3 # 行驶时间，秒


`2. 特定帧下,特定主车,以特定的加速度进行行驶的可视化`
    
    python ./python/main_visualize_data.py 0 \
    --start_timestamp 263300 # 帧数
    --track_id 145 # 主车id
    --duration 3 # 行驶时间，秒
    --acc 3.0 # 行驶的加速度

`3. 运行所有数据的碰撞检测，输出结果保存在collision_res_for_data000.txt中`

    python ./python/main_visualize_data.py 0 --duration 3 --get_all_collision_acc