from function import _calculate_distance


class TrajectoryManager:
    def __init__(self, max_disappeared=15, max_distance=50):
        self.trajectories = {}  # {track_id: {'last_box': box, 'cls': class, 'disappeared': 0, 'candidate_cls': None, 'candidate_count': 0}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, boxes, track_ids, clss):
        # 更新所有现有轨迹的消失状态并重置候选计数
        for track_id in list(self.trajectories.keys()):
            if track_id not in track_ids:
                self.trajectories[track_id]['disappeared'] += 1
                # 消失时重置候选类别计数
            if self.trajectories[track_id]['disappeared'] > self.max_disappeared/3:
                self.trajectories[track_id]['candidate_cls'] = None
                self.trajectories[track_id]['candidate_count'] = 0
                if self.trajectories[track_id]['disappeared'] > self.max_disappeared:
                    del self.trajectories[track_id]

        # 初始化输出类别列表（复制输入）
        output_clss = list(clss)

        for i, (box, track_id, current_cls) in enumerate(zip(boxes, track_ids, clss)):
            if track_id not in self.trajectories:
                # 新检测对象：尝试匹配消失轨迹
                best_match = None
                min_distance = float('inf')

                for tid, data in self.trajectories.items():
                    if tid not in track_ids:
                        dist = _calculate_distance(box, data['last_box'])
                        if dist < self.max_distance and dist < min_distance:
                            min_distance = dist
                            best_match = tid

                if best_match is not None:
                    # 重新分配ID
                    track_ids[i] = best_match
                    traj = self.trajectories[best_match]

                    # 更新轨迹数据
                    traj['last_box'] = box
                    traj['disappeared'] = 0

                    # 处理类别更新（使用候选机制）
                    stable_cls = traj['cls']
                    if stable_cls == 2.0:
                        # 稳定类别为2.0时，非2.0类别可直接覆盖
                        if current_cls != 2.0:
                            traj['cls'] = current_cls
                            output_clss[i] = current_cls
                        else:
                            output_clss[i] = stable_cls
                    else:
                        # 稳定类别为非2.0时
                        if current_cls == stable_cls:
                            # 相同类别：重置候选
                            traj['candidate_cls'] = None
                            traj['candidate_count'] = 0
                            output_clss[i] = stable_cls
                        elif current_cls != 2.0:
                            # 不同非2.0类别：更新候选
                            if traj['candidate_cls'] == current_cls:
                                traj['candidate_count'] += 1
                            else:
                                # 新候选类别，重置计数
                                traj['candidate_cls'] = current_cls
                                traj['candidate_count'] = 1

                            # 检查是否达到阈值
                            if traj['candidate_count'] >= self.max_disappeared:
                                traj['cls'] = current_cls
                                traj['candidate_cls'] = None
                                traj['candidate_count'] = 0
                                output_clss[i] = current_cls
                            else:
                                output_clss[i] = stable_cls
                        else:  # current_cls == 2.0
                            # 保持原类别
                            output_clss[i] = stable_cls
                else:
                    # 添加新轨迹（初始化候选状态）
                    self.trajectories[track_id] = {
                        'last_box': box,
                        'cls': current_cls,
                        'disappeared': 0,
                        'candidate_cls': None,
                        'candidate_count': 0
                    }
            else:
                # 更新现有轨迹
                traj = self.trajectories[track_id]
                stable_cls = traj['cls']

                # 更新轨迹数据
                traj['last_box'] = box
                traj['disappeared'] = 0

                # 处理类别更新（使用候选机制）
                if stable_cls == 2.0:
                    # 稳定类别为2.0时，非2.0类别可直接覆盖
                    if current_cls != 2.0:
                        traj['cls'] = current_cls
                        output_clss[i] = current_cls
                    else:
                        output_clss[i] = stable_cls
                else:
                    # 稳定类别为非2.0时
                    if current_cls == stable_cls:
                        # 相同类别：重置候选
                        traj['candidate_cls'] = None
                        traj['candidate_count'] = 0
                        output_clss[i] = stable_cls
                    elif current_cls != 2.0:
                        # 不同非2.0类别：更新候选
                        if traj['candidate_cls'] == current_cls:
                            traj['candidate_count'] += 1
                        else:
                            # 新候选类别，重置计数
                            traj['candidate_cls'] = current_cls
                            traj['candidate_count'] = 1

                        # 检查是否达到阈值
                        if traj['candidate_count'] >= self.max_disappeared:
                            traj['cls'] = current_cls
                            traj['candidate_cls'] = None
                            traj['candidate_count'] = 0
                            output_clss[i] = current_cls
                        else:
                            output_clss[i] = stable_cls
                    else:  # current_cls == 2.0
                        # 保持原类别
                        output_clss[i] = stable_cls

        return track_ids, output_clss


# from function import _calculate_distance
#
#
# class TrajectoryManager:
#     def __init__(self, max_disappeared=15, max_distance=50):
#         self.trajectories = {}  # {track_id: {'last_box': box, 'cls': class, 'disappeared': 0, 'candidate_cls': None, 'candidate_count': 0}}
#         self.max_disappeared = max_disappeared
#         self.max_distance = max_distance
#
#     def update(self, boxes, track_ids, clss):
#         # 更新所有现有轨迹的消失状态并重置候选计数
#         for track_id in list(self.trajectories.keys()):
#             if track_id not in track_ids:
#                 self.trajectories[track_id]['disappeared'] += 1
#                 # 消失时重置候选类别计数
#                 self.trajectories[track_id]['candidate_cls'] = None
#                 self.trajectories[track_id]['candidate_count'] = 0
#                 if self.trajectories[track_id]['disappeared'] > self.max_disappeared:
#                     del self.trajectories[track_id]
#
#         # 初始化输出类别列表（复制输入）
#         output_clss = list(clss)
#
#         for i, (box, track_id, current_cls) in enumerate(zip(boxes, track_ids, clss)):
#             if track_id not in self.trajectories:
#                 # 新检测对象：尝试匹配消失轨迹
#                 best_match = None
#                 min_distance = float('inf')
#
#                 for tid, data in self.trajectories.items():
#                     if tid not in track_ids:
#                         dist = _calculate_distance(box, data['last_box'])
#                         if dist < self.max_distance and dist < min_distance:
#                             min_distance = dist
#                             best_match = tid
#
#                 if best_match is not None:
#                     # 重新分配ID
#                     track_ids[i] = best_match
#                     traj = self.trajectories[best_match]
#
#                     # 更新轨迹数据
#                     traj['last_box'] = box
#                     traj['disappeared'] = 0
#
#                     # 处理类别更新（使用候选机制）
#                     stable_cls = traj['cls']
#                     if stable_cls == 2.0:
#                         # 稳定类别为2.0时，非2.0类别可直接覆盖
#                         if current_cls != 2.0:
#                             traj['cls'] = current_cls
#                             output_clss[i] = current_cls
#                         else:
#                             output_clss[i] = stable_cls
#                     else:
#                         # 稳定类别为非2.0时
#                         if current_cls == stable_cls:
#                             # 相同类别：重置候选
#                             traj['candidate_cls'] = None
#                             traj['candidate_count'] = 0
#                             output_clss[i] = stable_cls
#                         elif current_cls != 2.0:
#                             # 不同非2.0类别：更新候选
#                             if traj['candidate_cls'] == current_cls:
#                                 traj['candidate_count'] += 1
#                             else:
#                                 # 新候选类别，重置计数
#                                 traj['candidate_cls'] = current_cls
#                                 traj['candidate_count'] = 1
#
#                             # 检查是否达到阈值
#                             if traj['candidate_count'] >= self.max_disappeared:
#                                 traj['cls'] = current_cls
#                                 traj['candidate_cls'] = None
#                                 traj['candidate_count'] = 0
#                                 output_clss[i] = current_cls
#                             else:
#                                 output_clss[i] = stable_cls
#                         else:  # current_cls == 2.0
#                             # 保持原类别
#                             output_clss[i] = stable_cls
#                 else:
#                     # 添加新轨迹（初始化候选状态）
#                     self.trajectories[track_id] = {
#                         'last_box': box,
#                         'cls': current_cls,
#                         'disappeared': 0,
#                         'candidate_cls': None,
#                         'candidate_count': 0
#                     }
#             else:
#                 # 更新现有轨迹
#                 traj = self.trajectories[track_id]
#                 stable_cls = traj['cls']
#
#                 # 更新轨迹数据
#                 traj['last_box'] = box
#                 traj['disappeared'] = 0
#
#                 # 处理类别更新（使用候选机制）
#                 if stable_cls == 2.0:
#                     # 稳定类别为2.0时，非2.0类别可直接覆盖
#                     if current_cls != 2.0:
#                         traj['cls'] = current_cls
#                         output_clss[i] = current_cls
#                     else:
#                         output_clss[i] = stable_cls
#                 else:
#                     # 稳定类别为非2.0时
#                     if current_cls == stable_cls:
#                         # 相同类别：重置候选
#                         traj['candidate_cls'] = None
#                         traj['candidate_count'] = 0
#                         output_clss[i] = stable_cls
#                     elif current_cls != 2.0:
#                         # 不同非2.0类别：更新候选
#                         if traj['candidate_cls'] == current_cls:
#                             traj['candidate_count'] += 1
#                         else:
#                             # 新候选类别，重置计数
#                             traj['candidate_cls'] = current_cls
#                             traj['candidate_count'] = 1
#
#                         # 检查是否达到阈值
#                         if traj['candidate_count'] >= self.max_disappeared:
#                             traj['cls'] = current_cls
#                             traj['candidate_cls'] = None
#                             traj['candidate_count'] = 0
#                             output_clss[i] = current_cls
#                         else:
#                             output_clss[i] = stable_cls
#                     else:  # current_cls == 2.0
#                         # 保持原类别
#                         output_clss[i] = stable_cls
#
#         return track_ids, output_clss
