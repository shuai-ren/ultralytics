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
    '''
    判断点在多边形内, PNPoly算法
    '''
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


def detect():
    model = YOLO(opt.weights)
    names = model.names

    img = torch.zeros((1, 3, opt.img_size, opt.img_size))
    model(img)

    rtmpose = RTMPose("RTMPose/rtmpose-t-b3c5f0.onnx", device="cuda")

    data_deque = deque(maxlen=100)
    start(data_deque, opt.http_port, opt.rtsp_port, opt.mjpeg_port, out_size=(out_width, out_height), max_length=100)

    cap = cv2.VideoCapture(opt.video)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = 0
    reconnected = False
    # objects = []

    guidao_cls = 0
    person_cls = 1
    latest_guidao = None
    last_result_time = time.time()
    last_person_time = time.time()
    reset_guidao_time = 10
    update_guidao_time = 2
    ankle_thresh = 0.3
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
            # objects = []
            # results = model.track(frame, conf=opt.conf_thres, imgsz=opt.img_size, half=opt.half, persist=True)[0]
            results = model(frame, conf=opt.conf_thres, imgsz=opt.img_size, half=opt.half)[0]
            across = False

            if len(results) > 0:
                last_result_time = time.time()
                boxes = results.boxes
                cls = boxes.cls.int().tolist()
                # conf = boxes.conf.tolist()
                # xywhn = boxes.xywhn.tolist()
                # id = boxes.id.int().tolist()
                xyxy = boxes.xyxy.int().tolist()

                masks = results.masks
                # masks_data = masks.data
                masks_xy = masks.xy
                guidao_masks_xy = [masks for masks, c in zip(masks_xy, cls) if c == guidao_cls]

                person_bboxes = [bbox for bbox, c in zip(xyxy, cls) if c == person_cls]
                if person_bboxes:
                    last_person_time = time.time()
                    frame, all_kpts, all_scores = rtmpose.process(frame, person_bboxes)
                    # 0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳, 5-左肩, 6-右肩, 7-左肘, 8-右肘, 9-左手腕, 10-右手腕, 
                    # 11-左髋(臀部), 12-右髋(臀部), 13-左膝, 14-右膝, 15-左踝, 16-右踝
                    if latest_guidao is not None:
                        for kpts, scores in zip(all_kpts, all_scores):
                            lankle = scores[15] >= ankle_thresh
                            rankle = scores[16] >= ankle_thresh
                            if (lankle and pnpoly(latest_guidao, *kpts[15])) or (rankle and pnpoly(latest_guidao, *kpts[16])):
                                across = True
                else:
                    if guidao_masks_xy and (time.time() - last_person_time > update_guidao_time):
                        latest_guidao = guidao_masks_xy[0]
                        if opt.factor != 1:
                            center = np.mean(latest_guidao, axis=0)
                            centered_points = latest_guidao - center
                            cov_matrix = np.cov(centered_points, rowvar=False)
                            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                            pca_points = centered_points.dot(eigenvectors)
                            pca_points[:, 0] *= opt.factor
                            latest_guidao = pca_points.dot(eigenvectors.T) + center

                # for i in range(len(boxes)):
                #     obj = {"class_id": cls[i], 
                #         "name": names[cls[i]], 
                #         # "id": id[cls[i]],
                #         "confidence": conf[i],
                #         "relative_coordinates":{
                #             "center_x": xywhn[i][0],
                #             "center_y": xywhn[i][1], 
                #             "width": xywhn[i][2],
                #             "height": xywhn[i][3]},
                #     }

                #     objects.append(obj)

                # frame = results.plot()
                # frame = results.plot(masks=False, line_width=1)
                if latest_guidao is not None:
                    cv2.polylines(frame, [latest_guidao.astype("int32")], isClosed=True, color=colors(0, True), thickness=2)

            if time.time() - last_result_time > reset_guidao_time:
                latest_guidao = None

        json = {"frame_id": frame_id,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # "resolution": [w, h],
                # "objects": objects,
                "across": int(across)
        }
        data_deque.append((json, frame))

        print('fps: %.3f, \tavg_fps: %.3f, \tbuffer: %g' % (1 / (time.time() - t0), frame_id / (time.time() - start_time), len(data_deque)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='video url')
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--interval', type=int, default=2, help='sample interval')
    parser.add_argument('--http_port', type=int, required=True, help='http port')
    parser.add_argument('--rtsp_port', type=int, help='rtsp port')
    parser.add_argument('--mjpeg_port', type=int, help='mjpeg port')
    parser.add_argument('--factor', type=float, default=0.7, help='compression factor')

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


