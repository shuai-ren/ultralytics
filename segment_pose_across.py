'''
2024.9.3
'''

import os
os.environ["YOLO_VERBOSE"] = "false"
import argparse
import datetime
import time
from collections import deque

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from stream import start
from RTMPose.RTMPose import RTMPose


def pnpoly(verts, testx, testy):
    vertx = [xyvert[0] for xyvert in verts]
    verty = [xyvert[1] for xyvert in verts]
    nvert = len(verts)
    c = False
    j = nvert - 1
    for i in range(nvert):
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < (vertx[j]-vertx[i])*(testy-verty[i])/(verty[j]-verty[i])+vertx[i]):
            c = not c
        j = i
    return c


def calculate_intersection_ratio(contour1, contour2, img_shape):
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)

    cv2.fillPoly(mask1, [contour1], 255)
    cv2.fillPoly(mask2, [contour2], 255)

    intersection = cv2.bitwise_and(mask1, mask2)
    intersection_area = np.sum(intersection > 0)
    contour1_area = np.sum(mask1 > 0)
    ratio = intersection_area / contour1_area if contour1_area != 0 else 0

    return ratio


def detect():
    model = YOLO(opt.weights)
    names = model.names

    img = torch.zeros((1, 3, opt.img_size, opt.img_size))
    model(img)

    rtmpose = RTMPose(opt.rtmpose_weights, device="cuda")

    data_deque = deque(maxlen=100)
    start(data_deque, opt.json_port, opt.rtsp_port, opt.mjpeg_port, out_size=(out_width, out_height), max_length=100)

    cap = cv2.VideoCapture(opt.video)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = 0
    reconnected = False
    # objects = []

    cable_tray_idx = 0
    person_idx = 1
    fixed_cable_tray = None
    latest_cable_tray = None
    last_result_time = time.time()
    last_person_time = time.time()
    reset_cable_tray_time = 10
    update_cable_tray_time = 2
    kps_thresh = 0.5
    ratio_thresh = 0.8
    across = False

    start_time = time.time()
    while True:
        t0 = time.time()
        _, frame = cap.read()
        if frame is None:
            if reconnected:
                break
            cap.open(opt.video)
            reconnected = True
            continue

        frame_id += 1

        if frame_id % opt.interval == 0:
            results = model(frame, conf=opt.conf_thres, imgsz=opt.img_size, half=opt.half)[0]
            across = False

            if len(results) > 0:
                last_result_time = time.time()
                boxes = results.boxes
                cls = boxes.cls.int().tolist()
                xyxy = boxes.xyxy.int().tolist()
                person_bboxes = [bbox for bbox, c in zip(xyxy, cls) if c == person_idx]

                masks = results.masks
                cable_tray_contours = [contour for contour, c in zip(masks.xy, cls) if c == cable_tray_idx]

                if cable_tray_contours:
                    latest_cable_tray = cable_tray_contours[0]
                    if opt.factor != 1:
                        center = np.mean(latest_cable_tray, axis=0)
                        centered_points = latest_cable_tray - center
                        cov_matrix = np.cov(centered_points, rowvar=False)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                        pca_points = centered_points.dot(eigenvectors)
                        pca_points[:, 0] *= opt.factor
                        latest_cable_tray = pca_points.dot(eigenvectors.T) + center

                if person_bboxes:
                    last_person_time = time.time()
                    frame, all_kpts, all_scores = rtmpose.process(frame, person_bboxes)
                    print("rtmpose process fps: %.3f" % (1 / (time.time() - last_person_time)))
                    # 0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳, 5-左肩, 6-右肩, 7-左肘, 8-右肘, 9-左手腕, 10-右手腕, 
                    # 11-左髋(臀部), 12-右髋(臀部), 13-左膝, 14-右膝, 15-左踝, 16-右踝
                    if fixed_cable_tray is not None:
                        for kpts, scores in zip(all_kpts, all_scores):
                            lknee = scores[13] >= kps_thresh
                            rknee = scores[14] >= kps_thresh
                            lankle = scores[15] >= kps_thresh
                            rankle = scores[16] >= kps_thresh
                            if (lankle and pnpoly(fixed_cable_tray, *kpts[13])) or (rankle and pnpoly(fixed_cable_tray, *kpts[14])) \
                            or (lknee and pnpoly(fixed_cable_tray, *kpts[15])) or (rknee and pnpoly(fixed_cable_tray, *kpts[16])):
                                across = True
                                print("正在跨越")
                else:
                    if latest_cable_tray is not None and (time.time() - last_person_time > update_cable_tray_time):
                        fixed_cable_tray = latest_cable_tray.copy()

                if opt.debug:
                    if fixed_cable_tray is None and latest_cable_tray is not None:
                        fixed_cable_tray = latest_cable_tray.copy()

                if fixed_cable_tray is not None:
                    cv2.polylines(frame, [fixed_cable_tray.astype("int32")], isClosed=True, color=colors(0, True), thickness=2)
                if latest_cable_tray is not None:
                    cv2.polylines(frame, [latest_cable_tray.astype("int32")], isClosed=True, color=colors(8, True), thickness=2)
                    if fixed_cable_tray is not None:
                        ratio = calculate_intersection_ratio(latest_cable_tray.astype("int32"), fixed_cable_tray.astype("int32"), frame.shape[:2])
                        if ratio < ratio_thresh:
                            fixed_cable_tray = latest_cable_tray.copy()
                        cv2.putText(frame, f"RATIO: {ratio:.2f}", (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    latest_cable_tray = None
                if person_bboxes:
                    for box in person_bboxes:
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        cv2.rectangle(frame, p1, p2, colors(2, True), thickness=2, lineType=cv2.LINE_AA)


            if time.time() - last_result_time > reset_cable_tray_time:
                fixed_cable_tray = None

        json = {"frame_id": frame_id,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "across": int(across)
        }
        data_deque.append((json, frame))

        print('fps: %.3f, \tavg_fps: %.3f, \tbuffer: %g' % (1 / (time.time() - t0), frame_id / (time.time() - start_time), len(data_deque)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='video url')
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--rtmpose-weights', type=str, default="RTMPose/rtmpose-m.onnx", help='rtmpose weights path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--interval', type=int, default=1, help='sample interval')
    parser.add_argument('--json_port', type=int, required=True, help='json port')
    parser.add_argument('--rtsp_port', type=int, help='rtsp port')
    parser.add_argument('--mjpeg_port', type=int, help='mjpeg port')
    parser.add_argument('--factor', type=float, default=0.7, help='compression factor')
    parser.add_argument('--debug', action='store_true')

    opt = parser.parse_args()
    print(opt)

    import json

    cwd = os.getcwd()[7:]
    business_json_list = ["business.json", "/root/MBAB/AI/{}/etc/business.json".format(cwd)]
    out_width = 640
    out_height = 360
    add_line = lines = color = width = None
    for business_json in business_json_list:
        if os.path.isfile(business_json):
            business = json.load(open(business_json))
            params = business["business_params"]
            out_size = params.get("out_size")
            if out_size is not None:
                out_width = out_size.get("out_width")
                out_height = out_size.get("out_height")
                
            add_line = params.get("add_line")
            if add_line is not None:
                lines = add_line.get("lines")
                width = add_line.get("width")
                color = add_line.get("color")
            break
        
    detect()


