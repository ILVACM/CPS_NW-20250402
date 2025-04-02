import cv2

from pathlib import Path
import numpy as np
from typing import List

import torch
from shapely import Polygon

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


def clamp01(v: float):
    if v < 0:
        return 0
    elif v > 1:
        return 1
    else:
        return v


def lerp1d(a: float, b: float, t: float):
    if t <= 0:
        return a
    elif t >= 1:
        return b
    return a + (b - a) * clamp01(t)


def lerpnd(v1: List[float], v2: List[float], t: float) -> List[float]:
    if not v1 or not v2 or len(v1) != len(v2):
        raise ValueError("Lists a and b must have the same size and not be empty.")
    if t <= 0:
        return v1
    elif t >= 1:
        return v2
    return [lerp1d(a, b, t) for a, b in zip(v1, v2)]

def linear_move_toward_value(v1: float, v2: float, speed: float) -> float:
    if np.isclose(v1, v2):
        return v2
    dir = np.sign(v2 - v1)
    step = speed * abs(v2 - v1)
    val = v1 + dir * step
    if (dir > 0 and val > v2) or (dir < 0 and val < v2):
        return v2
    return val

def linear_move_toward_vector(v1: List[float], v2: List[float], speed: float) -> List[float]:
    start_pos = np.array(v1)
    end_pos = np.array(v2)
    dir = end_pos - start_pos
    dis = np.linalg.norm(dir)
    if np.isclose(dis, 0):
        return v2
    dir_norm = dir / dis
    current_pos = start_pos + dir_norm * speed
    if np.linalg.norm(current_pos - start_pos) > dis:
        return end_pos.tolist()
    return current_pos.tolist()


def expand_to_16_9(x1, y1, x2, y2):
    # 计算当前边界框的宽度和高度
    current_width = x2 - x1
    current_height = y2 - y1

    # 目标比例
    aspect_ratio = 16 / 9

    # 计算新的宽度和高度，使其保持16:9比例
    if current_width / current_height >= aspect_ratio:
        new_width = current_width
        new_height = current_width / aspect_ratio
    else:
        new_height = current_height
        new_width = current_height * aspect_ratio

    # 计算中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 计算新的边界框的左上角和右下角坐标
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)

    return new_x1, new_y1, new_x2, new_y2


# video process
# source_str:源文件夹 target_str:目标文件夹
def video_process(model, device, source_str: str, target_str: str, filename: str):
    # 视频输出处理
    input_path = Path(source_str) / filename
    output_path = increment_path(Path(target_str), exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening video file {input_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center = [(int)(frame_width / 2), (int)(frame_height / 2)]
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    delta_time = 1 / frame_rate  # second per frame用于处理帧间的平滑移动

    # 选择视频输出格式
    file_extension = filename.split('.')[-1].lower()
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 输出.mp4格式
    if file_extension == "avi":
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # 输出.avi格式
    elif file_extension == "mp4":
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 输出.avi格式
    out = cv2.VideoWriter(str(output_path / filename), fourcc, frame_rate, (frame_width, frame_height),
                          isColor=True)  # 创建一个VideoWriter对象用于写视频。

    # 预设常量
    line_thickness = 2  # OD框线宽
    screen_line_thickness = 6  # 屏幕框线宽
    screen_frame_color = (155, 248, 12)  # #9bf80c荧光绿（注意为BGR格式）
    center_point_color = (255, 191, 0) #蓝
    test_color = (0, 191, 255) #FFBF00琥珀黄
    screen_point_radius = 10  # 屏幕框中心点画圆半径
    screen_move_sensitivity = 2  # 屏幕框帧移动灵敏度
    screen_scale_sensitivity = 0.05  # 屏幕框帧缩放灵敏度
    screen_scale_threshold = 10  # 屏幕框缩放阈值，变化超过该阈值才触发屏幕框缩放
    height_scope = 2  # 高度范围（如height_scope=2，则屏幕框高度设为OD框平均高度的2倍）
    edge_padding_x = 50  # 屏幕框在x轴上固定的边缘填充
    edge_padding_y = 0  # 屏幕框在y轴上固定的边缘填充
    include_conf = 0.1  # 纳入屏幕框绘制考虑范围内的最小置信度
    edge_threshold = 0.1  # 超出的边缘阈值不考虑检测框（如1920*1080，则外圈的(192,108)像素不考虑）
    names = model.model.names
    vid_frame_count = 0  # 帧计数

    update_center_second = 2  # 当计算的中心点与当前中心点（屏幕框中心点）偏离的情况持续超过该间隔（s），摄像头发送云台移动控制信息
    update_distance_threshold = frame_width / 20  # 衡量计算的中心点与当前中心（屏幕框中心点）偏离的阈值
    update_center_frame_count = 0  # 计算的中心点与当前中心（屏幕框中心点）偏离的一定阈值的帧计数
    update_center_frame = frame_rate * update_center_second  # 将update_second换算成帧数：帧率 * 间隔（s）

    update_height_second = 2 # 当计算的画面高度与当前画面高度存在差距的情况持续超过该间隔（s），摄像头发送焦距调整控制信息
    update_height_threshold = frame_height / 5 # 衡量计算的画面高度与当前画面高度差距的阈值
    update_height_frame_count = 0 # 计算的高度与当前屏幕框高度相差一定阈值的帧计数
    update_height_frame = frame_rate * update_height_second # 将update_second换算成帧数：帧率 * 间隔（s）

    current_x1, current_y1, current_x2, current_y2 = (0, 0, frame_width, frame_height)
    current_center = frame_center  # 屏幕框中心点
    current_height = int(frame_height)  # 当前屏幕框高度
    lerp_center = frame_center
    lerp_height = int(frame_height)
    update_center = frame_center
    update_height = int(frame_height)
    linear_move_speed = 10 # 线性移动速度

    # 循环获取视频帧
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Extract the results
        results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", device=device)
        # 初始化
        latest_x1, latest_y1, latest_x2, latest_y2 = (0, 0, frame_width, frame_height)
        latest_center = [int((latest_x1 + latest_x2) / 2), int((latest_y1 + latest_y2) / 2)]
        latest_height = int(frame_height)
        detection_heights = []  # 最大的检测框高度

        if results[0].boxes.id is not None:
            if device == "cpu":
                print("inference using cpu")
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
            else:
                print("inference using gpu")
                boxes = results[0].boxes.xyxy.cuda()
                track_ids = results[0].boxes.id.int().cuda().tolist()
                clss = results[0].boxes.cls.cuda().tolist()
                confs = results[0].boxes.conf.cuda().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            center_points = []
            # 初始化最左上角和最右下角的坐标
            min_x, min_y = float(frame_width), float(frame_height)
            max_x, max_y = float(0), float(0)

            # 绘制bbox
            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                annotator.box_label(box, f" [{track_id}] {conf:.2f}", color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                # 对需要纳入绘制屏幕框、中心点考虑的检测框进行过滤/剔除
                # 当置信度小于一定程度时不纳入考虑
                if conf < include_conf:
                    continue
                # 当检测框位于画面边缘时不纳入考虑
                # x, y = bbox_center
                # x_threshold = frame_width * edge_threshold
                # y_threshold = frame_height * edge_threshold
                # if (x < x_threshold or x > (frame_width - x_threshold) or
                #         y < y_threshold or y > (frame_height - y_threshold)):
                #     continue

                center_points.append((float(bbox_center[0]), float(bbox_center[1])))
                # 更新最左上角和最右下角的坐标
                x1, y1, x2, y2 = box
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
                # 更新检测框高度
                detection_heights.append(y2 - y1)

            # 当没有正确初始化时，进行正确的初始化
            if int(min_x) == frame_width and int(min_y) == frame_height and int(max_x) == 0 and int(max_y) == 0:
                min_x, min_y, max_x, max_y = (0, 0, frame_width, frame_height)

            # 根据检测框高度取平均决定y轴上的边缘填充
            edge_padding_y = 0
            # 有检测目标时的处理
            if len(detection_heights) > 0:
                detection_heights_avg = sum(detection_heights) / len(detection_heights)
                edge_padding_y = detection_heights_avg * height_scope / 2

            x1, y1, x2, y2 = (min_x, min_y - edge_padding_y, max_x, max_y + edge_padding_y)

            # 处理最新屏幕框
            latest_x1, latest_y1, latest_x2, latest_y2 = expand_to_16_9(x1, y1, x2, y2)
            latest_center = [int((latest_x1 + latest_x2) / 2), int((latest_y1 + latest_y2) / 2)]
            latest_height = latest_y2 - latest_y1

        lerp_center = lerpnd(lerp_center, latest_center, delta_time * screen_move_sensitivity)
        lerp_center = [int(lerp_center[0]), int(lerp_center[1])]
        lerp_height = lerp1d(lerp_height, latest_height, delta_time * screen_scale_sensitivity)

        # 是否需要更新中心点、高度（模拟摄像头在中心点偏离一定距离的一定时间阈值后进行移动、缩放）
        lerp_center_v = np.array(lerp_center)
        current_center_v = np.array(current_center)
        latest_center_v = np.array(latest_center)
        lerp_current_center_distance = np.linalg.norm(lerp_center_v - current_center_v)
        lerp_latest_center_distance = np.linalg.norm(lerp_center_v - latest_center_v)
        current_lerp_height_difference = abs(current_height - lerp_height)
        latest_lerp_height_difference = abs(latest_height - lerp_height)

        # 如果目前中心点与平滑跟踪点距离远，进行帧计数
        if lerp_current_center_distance > update_distance_threshold:
            update_center_frame_count += 1
            if update_center_frame_count >= update_center_frame:
                update_center_frame_count = 0
                # 如果平滑跟踪点距离实时计算点近，取实时计算点（可以认为平滑跟踪点即将到达实时计算点，目前的实时计算点更接近理想中心点）
                if lerp_latest_center_distance < update_distance_threshold:
                    update_center = latest_center
                # 如果平滑跟踪点距离实时计算点远，取平滑跟踪点到实时计算点的中点（可以认为画面出现较大变化，实时计算点发生跳跃，平滑跟踪点即将向实时计算点移动，取中点为理想中心点）
                else:
                    update_center = ((lerp_center[0] + latest_center[0]) / 2, (lerp_center[1] + latest_center[1]) / 2)
        # 如果目前中心点与平滑跟踪点距离近，可以认为短时间内还不需要更新中心点
        else:
            update_center_frame_count = 0

        # 如果目前画面大小（屏幕框大小）与平滑屏幕框大小的高度差距大，进行帧计数
        if current_lerp_height_difference > update_height_threshold:
            update_height_frame_count += 1
            if update_center_frame_count >= update_height_frame:
                update_height_frame_count = 0
                if latest_lerp_height_difference < update_height_threshold:
                    update_height = latest_height
                else:
                    update_height = (lerp_height + latest_height) / 2
        else:
            update_height_frame_count = 0

        # 屏幕框中心点线性插值
        current_center = linear_move_toward_vector(current_center, update_center, linear_move_speed)
        current_center = [int(current_center[0]), int(current_center[1])]
        # current_center = lerpnd(current_center, update_center, delta_time * screen_move_sensitivity)
        # current_center = [int(current_center[0]), int(current_center[1])]

        # 屏幕框线性插值
        # current_height = lerp1d(current_height, update_height, delta_time * screen_scale_sensitivity)
        current_height = linear_move_toward_value(current_height, update_height, screen_scale_sensitivity)
        x1, y1, x2, y2 = (current_center[0] - edge_padding_x, current_center[1] - current_height / 2,
                          current_center[0] + edge_padding_x, current_center[1] + current_height / 2)
        current_x1, current_y1, current_x2, current_y2 = expand_to_16_9(x1, y1, x2, y2)

        # 绘制平滑屏幕框
        poly = Polygon([(current_x1, current_y1), (current_x2, current_y1),
                        (current_x2, current_y2), (current_x1, current_y2)])
        polygon_coords = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(frame, [polygon_coords], isClosed=True, color=test_color,
                      thickness=screen_line_thickness)

        # 绘制边缘范围
        # x_threshold = frame_width * edge_threshold
        # y_threshold = frame_height * edge_threshold
        # poly = Polygon([(x_threshold, y_threshold), (frame_width - x_threshold, y_threshold),
        #                 (frame_width - x_threshold, frame_height - y_threshold),
        #                 (x_threshold, frame_height - y_threshold)])
        # polygon_coords = np.array(poly.exterior.coords, dtype=np.int32)
        # cv2.polylines(frame, [polygon_coords], isClosed=True, color=(255, 255, 255),
        #               thickness=line_thickness)

        # 是否需要缩小
        need_zoom_out = (current_height > frame_height)
        # 如果没有检测目标，作出决策缩小画面 TODO:如果没有检测目标，有距离过远导致目标太小检测不到 和 距离太近需要拉远画面 两种情况
        if len(detection_heights) <= 0:
            need_zoom_out = True

        # 绘制文字信息（屏幕框偏偏移量、放缩决策、屏幕框中心）
        x_offset = latest_center[0] - current_center[0]
        y_offset = latest_center[1] - current_center[1]
        info_label = f"x_offset:{x_offset},y_offset:{y_offset},zoom_out:{need_zoom_out}"
        text_size, _ = cv2.getTextSize(
            info_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
        )
        text_x = 10
        text_y = text_size[1] + 10
        cv2.putText(
            frame, info_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, screen_frame_color, line_thickness
        )
        # 绘制实时计算的中心点
        cv2.circle(frame, latest_center, screen_point_radius, screen_frame_color, thickness=-1)
        # 平滑点
        cv2.circle(frame, lerp_center, screen_point_radius, center_point_color, thickness=-1)
        # 屏幕框中心点
        cv2.circle(frame, current_center, screen_point_radius, test_color, thickness=-1)

        # 实时推理
        # cv2.imshow("Test", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # 保存
        out.write(frame)

        vid_frame_count += 1

    # Release the video writer
    out.release()
    # Release the video capture
    cap.release()


if __name__ == "__main__":
    # 输入文件夹
    source_str = "./datasets"
    # 输出文件夹
    target_str = "./track_outputs_test/exp"
    filename = "clip_6.mp4"
    model = YOLO('yolov8n.pt')
    device = "cpu" # 使用cpu推理
    model.to(device)
    video_process(model, device, source_str, target_str, filename)

    # 批量处理
    # filelist = ["1", "2", "3", "4", "5", "7", "8"]
    # file_head = "clip_"
    # file_tail = ".mp4"
    # for i in range(len(filelist)):
    #     filename = file_head + filelist[i] + file_tail
    #     video_process(source_str, target_str, filename)
