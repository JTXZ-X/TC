import cv2
import numpy as np


def draw_region_info(frame: np.ndarray, region, line_thickness=2, region_thickness=2):
    """
    在视频帧上绘制计数区域信息

    参数:
        frame: 要绘制的视频帧
        region: 包含区域信息的字典，包含:
            counts - 显示的数字
            region_color - 区域颜色
            text_color - 文本颜色
            polygon - 区域的Polygon对象
        line_thickness: 文本和轮廓的线宽
        region_thickness: 区域边界的线宽
    """
    region_label = str(region["counts"])
    region_color = region["region_color"]
    text_color = region["text_color"]
    polygon = region["polygon"]

    # 准备多边形坐标
    polygon_coords = np.array(polygon.exterior.coords, dtype=np.int32)
    # 获取中心点并转换为整数坐标
    centroid = polygon.centroid
    center_x, center_y = int(centroid.x), int(centroid.y)
    # 获取文本尺寸
    text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=line_thickness)
    # 计算文本位置
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    # 计算文本框坐标（带5像素填充）
    (left, top, right, bottom) = (
        text_x - 5,
        text_y - text_size[1] - 5,
        text_x + text_size[0] + 5,
        text_y + 5
    )

    # 绘制文本框背景（实心矩形）
    cv2.rectangle(frame, (left, top), (right, bottom), region_color, -1)
    # 绘制文本
    cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, line_thickness)
    # 绘制多边形轮廓
    cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)


def display_total_counts(frame, counts, start_x, start_y, y_offset=50, font_scale=1.3, color=(0, 0, 0), thickness=3):
    """
    在视频帧右上角显示鱼类计数统计信息

    参数:
        frame: 要绘制的视频帧
        counts: 包含统计数据的字典
        start_x: 文本起始X坐标
        start_y: 文本起始Y坐标
        y_offset: 每行文本的垂直间距(默认50)
        font_scale: 字体缩放比例(默认1.3)
        color: 文本颜色(默认黑色)
        thickness: 文本厚度(默认3)
    """
    # 要显示的数据类型及其标签
    count_labels = [
        ("Total Count", "total"),
        ("Yellowfin Count", "yellowfin"),
        ("Albacore Count", "albacore"),
        ("Bigeye Count", "bigeye"),
        ("Skipjack Count", "skipjack"),
        ("Others Count", "others"),
        ("Uncertainty Count", "uncertainty")
    ]

    # 循环显示每类计数
    for i, (label, key) in enumerate(count_labels):
        # 计算当前行Y坐标
        current_y = start_y + i * y_offset
        # 格式化文本
        text = f"{label}: {counts[key]}"
        # 绘制文本
        cv2.putText(frame, text, (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
