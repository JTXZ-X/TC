import numpy as np
from shapely import Point, Polygon


def _calculate_distance(box1, box2):
    # 计算两个框中心点的欧氏距离
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def check_direction(current_pos: Point, last_pos: Point, region: Polygon):
    if last_pos is None:
        return None

    # 判断进入区域
    if not region.contains(last_pos) and region.contains(current_pos):
        return "entry"

    # 判断离开区域
    if not region.contains(current_pos) and region.contains(last_pos):
        return "exit"

    return None


def determining_counting(counts: dict, direction: str, clss: str):
    multiplier = 1 if direction == "exit" else -1

    if direction in ['entry', 'exit']:
        counts['total'] += multiplier
        counts[clss] += multiplier
