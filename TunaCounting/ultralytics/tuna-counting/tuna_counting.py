import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from improved_tracking import TrajectoryManager
from function import _calculate_distance, check_direction, determining_counting
from draw import draw_region_info, display_total_counts

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(900, 230), (1920, 230), (1920, 850), (900, 850)]),
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),
        "text_color": (0, 0, 0),
        "edges": ["right", "top", "left", "bottom"],
        "last_positions": {},
    },
]


def mouse_callback(event, x, y, flags, param):
    global current_region
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
        weights="../../ultralytics/yolov8m.pt",
        source=None,
        device="cpu",
        view_img=False,
        save_img=False,
        exist_ok=False,
        classes=None,
        track_thickness=2,
        counts=None
):
    vid_frame_count = 0

    object_sizes = {}  # Dictionary to store sizes by object ID
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")
    names = model.model.names

    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    save_dir = Path("E:/金枪鱼识别计数测试结果/2.Ours-nano")
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    trajectory_manager = TrajectoryManager(max_disappeared=30, max_distance=300)

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # 获取模型预测结果
        results = model.track(frame, persist=True, classes=classes, tracker="botsort.yaml")

        # 当前识别到了对象
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # print("\ntrack_ids = {}\n".format(track_ids))
            clss = results[0].boxes.cls.cpu().tolist()

            # 延续id, 并更新类别
            track_ids, clss = trajectory_manager.update(boxes, track_ids, clss)
            # print(f"track_ids = {track_ids}, clss = {clss}")

            annotator = Annotator(frame, line_width=3, example=str(names), font_size=16)

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                # (box[0], box[1])是左上角点, (box[2], box[3])是右下角点
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                object_clss = names[cls]

                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # current_pos、last_pos以及direction的判定
                for region in counting_regions:
                    # 确定计数条件
                    current_pos = Point(bbox_center)
                    last_pos = region["last_positions"].get(track_id, None)
                    direction = check_direction(current_pos=current_pos, last_pos=last_pos, region=region["polygon"])

                    # print("current_pos = {}, last_pos = {}, direction = {}".format(current_pos, last_pos, direction))
                    # 根据计数条件判定计数
                    determining_counting(counts, direction, object_clss)

                    # 将当前帧的中心点设为下一帧的last_positions
                    region["last_positions"][track_id] = current_pos

                    if region["polygon"].contains(current_pos):
                        region["counts"] += 1

        for region in counting_regions:
            draw_region_info(frame, region)

        # 显示右上角的总离开计数
        frame_width = frame.shape[1]
        start_x = frame_width - 1900  # 右上角位置
        display_total_counts(frame, counts, start_x=start_x, start_y=50)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Improved Region Counting")
                cv2.setMouseCallback("Improved Region Counting", mouse_callback)
            cv2.imshow("Improved Region Counting", frame)

        if save_img:
            video_writer.write(frame)

        # 每帧后重置区域计数（根据需要）
        for region in counting_regions:
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-img", default=True, action="store_true", help="show results")
    parser.add_argument("--save-img", default=True, action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--weights", type=str,
                        default="../runs/detect/C2f_RepGhosts/weights/best.pt",
                        help="initial weights path")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source",
                        default=r"./Demo.mp4",
                        type=str, help="video file path")
    return parser.parse_args()


def main(opt):
    counts = {
        'total':        0,
        'yellowfin':    0,
        'albacore':     0,
        'bigeye':       0,
        'skipjack':     0,
        'others':       0,
        'uncertainty':  0,
    }
    """Main function."""
    run(**vars(opt), counts=counts)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
