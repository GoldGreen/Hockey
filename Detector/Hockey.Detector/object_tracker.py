import json
import pika
import math
from collections import Counter
import os
import time
import tensorflow as tf
from collections import namedtuple
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import webcolors
from tensorflow.python.ops.gen_math_ops import angle, atan2, sqrt
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_args
from tensorflow.python.framework.ops import control_dependencies
from numpy.lib.function_base import average
from typing import Match
from functools import WRAPPER_UPDATES
import sys
import inspect

import requests


class HockeyResult(object):
    pass


class HockeyObject(object):
    pass


class VideoObject(object):
    pass


def get_drawed_field(frame, start_x, end_x, start_y, end_y, width_d, height_d):

    field = frame[start_y: end_y, start_x: end_x]
    field[:] = (108, 124, 142)

    cv2.line(field, (int(width_d/2), 0),
             (int(width_d/2), height_d), (169, 32, 62), 3)

    cv2.line(field, (int(2*width_d/5), 0),
             (int(2*width_d/5), height_d), (80, 149, 182), 3)

    cv2.line(field, (int(3*width_d/5), 0),
             (int(3*width_d/5), height_d), (80, 149, 182), 3)

    cv2.line(field, (int(1*width_d/10), 0),
             (int(1*width_d/10), height_d), (169, 32, 62), 2)

    cv2.line(field, (int(9*width_d/10), 0),
             (int(9*width_d/10), height_d), (169, 32, 62), 2)

    cv2.ellipse(field, (int(width_d/2), int(height_d/2)), (23, 23), 0,
                0, 360, (80, 149, 182), 2)

    cv2.ellipse(field, (int(4*width_d/5), int(4*height_d/5)), (23, 23), 0,
                0, 360, (169, 32, 62), 2)

    cv2.ellipse(field, (int(4*width_d/5), int(1*height_d/5)), (23, 23), 0,
                0, 360, (169, 32, 62), 2)

    cv2.ellipse(field, (int(1*width_d/5), int(4*height_d/5)),  (23, 23), 0,
                0, 360, (169, 32, 62), 2)

    cv2.ellipse(field, (int(1*width_d/5), int(1*height_d/5)),  (23, 23), 0,
                0, 360, (169, 32, 62), 2)

    cv2.ellipse(field, (int(1*width_d/10), int(height_d / 2)),
                (20, 20), 0, 270, 450, (80, 149, 182), 2)

    cv2.ellipse(field, (int(9*width_d/10), int(height_d / 2)),
                (20, 20), 0, 90, 270, (80, 149, 182), 2)

    return field


def colorLen(first, second):
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2

# x_src,y_src - from src (3D) img, return [x,y]


def perspectPoingTransormation(x_src, y_src, h):
    # Calculate transfom matrix (h)
    # h, status = cv2.findHomography(pts_src, pts_dst)
    # for cv2.perspectiveTransform need this form
    pts = np.array([[[x_src, y_src]]], dtype="float32")
    arr = cv2.perspectiveTransform(pts, h)
    return np.array(arr).reshape(2)


def create_circular_mask(h, w):
    mask = np.zeros([h, w, 3], dtype=np.uint8)
    mask_h, mask_w = mask.shape[:2]
    center = (int(mask_w/2), int(2*mask_h/5))

    len = int(min(mask_w, mask_h)/3)
    axes = (len, len)
    color = (255, 255, 255)
    cv2.ellipse(mask, center, axes, 0, 0, 360, color, thickness=-1)

    return np.mean(mask, axis=2) == 255


def closest_colour(requested_colour):
    min_colours = {}
    # ЗАВИСИТ ОТ ВЕРСИИ БИБЛЫ, МБ ИМЯ ТОЛЬКО ЭТОЙ ФУНКЦИИ КАПСОМ
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def most_frequent(List):
    return max(set(List), key=List.count)


def sign(num):
    return -1 if num < 0 else 1


def pLen(first, second):
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def getLineKB(p1, p2):
    k = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p2[1] - k*p2[0]
    return (k, b)


def findPointWithMinDist(points, firstP, secondP):
    res = None

    for point in points:
        if res == None:
            res = point
        else:
            resLen = pLen(res, firstP) - pLen(res, secondP)
            pointLen = pLen(point, firstP) - pLen(point, secondP)

            if abs(pointLen) < abs(resLen):
                res = point

    return res


def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False):  # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', 'checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov3', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'aboba',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', "data/output/result.mp4",
                    'path to output video')
flags.DEFINE_string('outputMap', "data/output/resultMap.mp4",
                    'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.35, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean(
    'info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean(
    'count', False, 'count objects being tracked on screen')


def main(_argv):
    headers = {"content-type": "application/json"}
    os.chdir(get_script_dir())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # firstCommandColor = (217, 201, 56)
    # secondCommandColor = (204, 204, 204)

    firstCommandColor = (41, 143, 183)
    secondCommandColor = (214, 214, 143)

    def unique_count_app(roi):
        h, w = roi.shape[:2]
        firstCount = 0
        secondCount = 0
        if(h > 0 and w > 0):
            mask = create_circular_mask(h, w)
            maskShaped = roi[mask]
            # roi[~mask] = (0, 0, 0)

            colors, count = np.unique(
                maskShaped, axis=0, return_counts=True)

            for color in colors:
                firstLen = colorLen(color, firstCommandColor)
                secondLen = colorLen(color, secondCommandColor)
                if firstLen < maxLen and firstLen < secondLen:
                    firstCount += 1
                elif secondLen < maxLen:
                    secondCount += 1

        return firstCommandColor if firstCount > secondCount else secondCommandColor

    maxLen = 30000

    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load(
        FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoResponce = VideoObject()
    videoResponce.path = video_path
    videoResponce.framesCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    videoResponce.width = width
    videoResponce.height = height

    requests.put("http://localhost:5000/hockey/start",
                 data=json.dumps(videoResponce, default=lambda o: o.__dict__), headers=headers)

    out = None
    outMap = None

    cameraDistance = 100

    if FLAGS.output:
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        outMap = cv2.VideoWriter(
            FLAGS.outputMap, codec, fps, (415, int(416 * height/width)))

    frame_num = 0
    heatMapWidth = 20
    heatMapHeight = 15

    heatMapMatrix = [[0 for _ in range(heatMapWidth)]
                     for _ in range(heatMapHeight)]

    heatMapChangedMatrix = [[False for _ in range(heatMapWidth)]
                            for _ in range(heatMapHeight)]

    maxMemoryWay = 20
    heatMapMemoryMatrix = [[0 for _ in range(heatMapWidth)]
                           for _ in range(heatMapHeight)]
    addedHeatValue = 20

    while True:
        return_value, frame = vid.read()
        angleFrame = None
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameCopy = frame.copy()
            # image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        angleFrame = frame.copy()
        trashSize = 416
        trashWidth = 416
        trashHeight = int(trashSize * height/width)

        angleFrame = cv2.resize(
            angleFrame, (trashSize, trashHeight))
        angleFrame = cv2.cvtColor(angleFrame, cv2.COLOR_RGB2GRAY)

        ret, angleFrame = cv2.threshold(
            angleFrame, 180, 220, cv2.THRESH_BINARY)

        miniAngleFrame = cv2.resize(
            angleFrame, (100, int(100 * height/width)))

        miniAngleFrame = cv2.resize(
            miniAngleFrame, (trashWidth, trashHeight))

        miniAngleFrame = cv2.medianBlur(miniAngleFrame, 61)

        contours, _ = cv2.findContours(
            miniAngleFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        angleFrame = cv2.cvtColor(angleFrame, cv2.COLOR_GRAY2RGB)

        maxLen = max(map(lambda x: len(x), contours))
        contours = list(filter(lambda x: len(x) == maxLen, contours))

        points = [
            (point[0], point[1])
            for hierarhy in contours
            for contour in hierarhy
            for point in contour
        ]

        minX = 100000
        minXIndex = -1

        maxX = -1
        maxXIndex = -1

        for i, point in enumerate(points):
            if point[0] < minX:
                minX = point[0]
                minXIndex = i
            elif point[0] == minX:
                if point[1] > points[minXIndex][1]:
                    minXIndex = i
            elif point[0] > maxX:
                maxX = point[0]
                maxXIndex = i
            elif point[0] == maxX:
                if point[1] > points[maxXIndex][1]:
                    maxXIndex = i

        minY = 1000
        minYIndex = -1

        for i, point in enumerate(points):
            if point[1] < minY:
                minY = point[1]
                minYIndex = i

        higherY = minY + 20

        minP = (points[minXIndex][0], points[minXIndex][1])
        maxP = (points[maxXIndex][0], points[maxXIndex][1])

        closestPPoints = list(filter(lambda x: x[1] < higherY, points))
        closestP = findPointWithMinDist(closestPPoints, minP, maxP)

        minPIndex = points.index(minP)
        maxPIndex = points.index(maxP)

        rightUpPoints = list(filter(lambda x: x[0] >
                                    maxPIndex, enumerate(points)))
        rightUpPoints = list(map(lambda x: x[1], rightUpPoints))

        leftUpPoints = list(filter(lambda x: x[0] <
                                   minPIndex, enumerate(points)))
        lefttUpPoints = list(map(lambda x: x[1], leftUpPoints))

        closestRightUpP = findPointWithMinDist(rightUpPoints, closestP, maxP)
        closestLeftUpP = findPointWithMinDist(lefttUpPoints, closestP, minP)

        heightDelta = 15
        closestP = (closestP[0], closestP[1] + heightDelta)
        closestRightUpP = (
            closestRightUpP[0], closestRightUpP[1] + heightDelta)
        closestLeftUpP = (closestLeftUpP[0], closestLeftUpP[1] + heightDelta)

        cv2.drawContours(angleFrame, contours, -1,
                         (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

        for point in [minP, maxP, closestP, closestRightUpP, closestLeftUpP]:
            cv2.ellipse(angleFrame, point, (20, 20),
                        0, 0, 360, (0, 255, 0), -1)

        angle = math.atan2(points[maxXIndex][1] - points[minXIndex]
                           [1], points[maxXIndex][0] - points[minXIndex][0])

        cv2.putText(angleFrame, str(math.degrees(angle)),
                    (0, 50), 0, 0.75, (255, 0, 0), 2)

        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]

        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        pred_bbox = [bboxes, scores, classes, num_objects]
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = list(class_names.values())

        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# detection
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])

        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
# end detection

# correct position
# end correct position

# tracking
        tracker.predict()
        tracker.update(detections)

        result = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            currColor = unique_count_app(roi)
            t = namedtuple("t", ("track", "bbox", "currColor"))
            result.append(t(track, bbox, currColor))
# end tracking

# draw objects
        for t_temp in result:
            color = colors[int(t_temp.track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(t_temp.bbox[0]), int(t_temp.bbox[1])), (int(
                t_temp.bbox[2]), int(t_temp.bbox[3])), color, 2)

            actual_name, closest_name = get_colour_name(t_temp.currColor)
            string = "{}-{} color: {}".format(t_temp.track.get_class(),
                                              t_temp.track.track_id, closest_name)

            cv2.rectangle(frame, (int(t_temp.bbox[0]), int(t_temp.bbox[1] - 30)),
                          (int(t_temp.bbox[0]) + len(string) * 13, int(t_temp.bbox[1])), color, -1)

            cv2.putText(frame, string, (int(t_temp.bbox[0]), int(
                t_temp.bbox[1] - 10)), 0, 0.75, (50, 10, 24), 2)

        widthK = (width/trashWidth)
        heightK = (height/trashHeight)

        cleanClosestP = (closestP[0] * widthK, closestP[1] * heightK)
        cleanMinP = (minP[0] * widthK, minP[1] * heightK)
        cleanMaxP = (maxP[0] * widthK, maxP[1] * heightK)
        cleanRightUpP = (closestRightUpP[0] *
                         widthK, closestRightUpP[1] * heightK)
        cleanLeftUpP = (closestLeftUpP[0] *
                        widthK, closestLeftUpP[1] * heightK)

        cleanClosestP = (int(cleanClosestP[0]), int(cleanClosestP[1]))
        cleanMinP = (int(cleanMinP[0]), int(cleanMinP[1]))
        cleanMaxP = (int(cleanMaxP[0]), int(cleanMaxP[1]))
        cleanRightUpP = (int(cleanRightUpP[0]), int(cleanRightUpP[1]))
        cleanLeftUpP = (int(cleanLeftUpP[0]), int(cleanLeftUpP[1]))

        for point in [cleanClosestP, cleanMinP, cleanMaxP, cleanRightUpP, cleanLeftUpP]:
            cv2.ellipse(frame, point, (50, 50), 0,
                        0, 360, (255, 0, 0), -1)
# end draw objects

# draw field
        width_d = 400
        height_d = 220

        field = get_drawed_field(frame, 0, width_d, 0,
                                 height_d, width_d, height_d)

        cameraPos = (int(width_d/2), int(height_d*1.8))
        cameraAngle = 65
        cameraAngle = math.radians(cameraAngle)

        angle *= 1.9

        cameraAngleStart = cameraAngle / 2 + angle
        cameraAngleEnd = cameraAngle / 2 - angle

        cameraAngleStartX = height_d * math.tan(cameraAngleStart)
        cameraAngleEndX = height_d * math.tan(cameraAngleEnd)

        cameraAngleXLeft = width_d / 2 - cameraAngleStartX
        cameraAngleXRight = width_d / 2 + cameraAngleEndX

        (k1, b1) = getLineKB((cameraAngleXLeft, 0), cameraPos)
        (k2, b2) = getLineKB((cameraAngleXRight, 0), cameraPos)

        x1 = int((height_d - b1) / k1)
        x2 = int((height_d - b2) / k2)

        cameraAngleXLeft = int(cameraAngleXLeft)
        cameraAngleXRight = int(cameraAngleXRight)

        cameraLeft = (x1, height_d)
        cameraRight = (x2, height_d)

        if cameraRight[0] > width_d:
            cameraRight = (width_d, cameraRight[1])

        if cameraLeft[0] < 0:
            cameraLeft = (0, cameraLeft[1])

        closestMiniP = findPointWithMinDist(
            map(lambda i: (i, 0), range(width_d)), cameraLeft, cameraRight)

        closestLeftUpCamera = (int((0 - b1) / k1), 0)
        closestRightUpCamera = (int((0 - b2) / k2), 0)

        if closestLeftUpCamera[0] < 0:
            closestLeftUpCamera = (0, 0)

        if closestRightUpCamera[0] > width_d:
            closestRightUpCamera = (width_d, 0)

        miniMapArray = np.array(
            [[cameraLeft[0], cameraLeft[1]],
             [cameraRight[0], cameraRight[1]],
             [closestLeftUpCamera[0], closestLeftUpCamera[1]],
             [closestRightUpCamera[0], closestRightUpCamera[1]],
             [closestMiniP[0], closestMiniP[1]]], np.float32)

        sourceArray = np.array(
            [[cleanMinP[0], cleanMinP[1]],
             [cleanMaxP[0], cleanMaxP[1]],
             [cleanLeftUpP[0], cleanLeftUpP[1]],
             [cleanRightUpP[0], cleanRightUpP[1]],
             [cleanClosestP[0], cleanClosestP[1]]], np.float32)

        h, _ = cv2.findHomography(sourceArray, miniMapArray)

        drawedPerspectve = cv2.warpPerspective(
            frameCopy, h, (width_d, height_d))

        cv2.imshow("drawedPerspectve", drawedPerspectve)


# heat map
        for i in range(heatMapHeight):
            for j in range(heatMapWidth):
                heatMapChangedMatrix[i][j] = False

        for t_temp in result:
            # Добавить рассчёт центра по точке полученной из функции преобразования (точка на матрицу перспективы)
            center = (t_temp.bbox[1] + (t_temp.bbox[3] - t_temp.bbox[1]) / 2,
                      t_temp.bbox[0] + (t_temp.bbox[2] - t_temp.bbox[0]) / 2)

            center = perspectPoingTransormation(
                center[1], center[0], h)

            center = (center[1] / height_d, center[0] / width_d)

            if t_temp.track.get_class() != "ref" and center[0] >= 0 and center[0] < 1 and center[1] >= 0 and center[1] < 1:
                center = (int(center[0] * heatMapHeight),
                          int(center[1] * heatMapWidth))

                heatMapChangedMatrix[center[0]][center[1]] = True

                if heatMapMatrix[center[0]][center[1]] < 256:
                    heatMapMatrix[center[0]][center[1]] += addedHeatValue

        for i in range(heatMapHeight):
            for j in range(heatMapWidth):
                if heatMapChangedMatrix[i][j]:
                    heatMapMemoryMatrix[i][j] = 0
                else:
                    heatMapMemoryMatrix[i][j] += 1

        for i in range(heatMapHeight):
            for j in range(heatMapWidth):
                if heatMapMemoryMatrix[i][j] > maxMemoryWay:
                    heatMapMatrix[i][j] -= addedHeatValue
                    heatMapMemoryMatrix[i][j] -= 1

        for i in range(heatMapHeight):
            for j in range(heatMapWidth):
                if heatMapMatrix[i][j] < 0:
                    heatMapMatrix[i][j] = 0
# end heat map

# end draw field

        jsonRes = HockeyResult()
        jsonRes.frameNum = frame_num
        jsonRes.players = []

        for t_temp in result:

            transformedcoordinate = perspectPoingTransormation(
                t_temp.bbox[0], t_temp.bbox[1], h)

            center = (int(transformedcoordinate[0]),
                      int(transformedcoordinate[1]))

            jsonObj = HockeyObject()
            jsonObj.id = t_temp.track.track_id
            jsonObj.bbox = [t_temp.bbox[0], t_temp.bbox[1]]

            jsonObj.color = (255, 255, 255) if t_temp.track.get_class(
            ) == "ref" else t_temp.currColor

            jsonObj.type = t_temp.track.get_class()
            jsonObj.center = center

            jsonRes.players.append(jsonObj)

            cv2.ellipse(field, center, (10, 10), 0,
                        0, 360, (255, 0, 0) if t_temp.track.get_class() == "ref" else t_temp.currColor, -1)

            cv2.ellipse(field, center, (12, 12), 0,
                        0, 360, (0, 0, 0), 4)

            if t_temp.track.get_class() != "ref":
                cv2.putText(field, str(t_temp.track.track_id),
                            (center[0] - 4, center[1] + 4), 0, 0.45, (255, 0, 0), 2)

        requests.put("http://localhost:5000/hockey/frame",
                     data=json.dumps(jsonRes, default=lambda o: o.__dict__), headers=headers)

        for point in [closestMiniP, cameraLeft, cameraRight, closestLeftUpCamera, closestRightUpCamera]:
            cv2.ellipse(field, point, (15, 15), 0,
                        0, 360, (255, 0, 0), -1)

        cv2.line(field, cameraPos,
                 (cameraAngleXLeft, 0), (255, 0, 0), 2)

        cv2.line(field, cameraPos,
                 (cameraAngleXRight, 0), (0, 0, 255), 2)

        cv2.rectangle(field, (0, 0),
                      (width_d, height_d), (0, 33, 55), 3)

# draw heat map
        heatMap = get_drawed_field(frame, original_w - width_d,
                                   original_w, 0, height_d,
                                   width_d, height_d)

        heatMapDrawingWidth = int(width_d / heatMapWidth)
        heatMapDrawingHeight = int(height_d / heatMapHeight)

        for (i, heatMapArray) in enumerate(heatMapMatrix):
            for (j, value) in enumerate(heatMapArray):
                if(value != 0):
                    pt1 = (j*heatMapDrawingWidth, i*heatMapDrawingHeight)
                    pt2 = (pt1[0] + heatMapDrawingWidth,
                           pt1[1] + heatMapDrawingHeight)

                    cv2.rectangle(heatMap, pt1, pt2,
                                  (value, 0, 255 - value), -1
                                  )

        cv2.rectangle(heatMap, (0, 0),
                      (width_d, height_d), (0, 33, 55), 3)
# end draw heat map

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        angleResult = cv2.cvtColor(angleFrame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
            cv2.imshow("Angle", angleResult)

        if FLAGS.output:
            out.write(result)
            outMap.write(angleResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
