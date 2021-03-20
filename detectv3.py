import argparse
from sys import platform
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from shapely.geometry import Point, Polygon
import cv2
import geopandas as gpd


def non_max_suppression_fast(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > iou_threshold)[0])))
    new_boxes = xyxy2xywh(boxes[pick])
    return new_boxes


def load_geographic_data(x):
    # input will be xyxy, generated from non_max_suppression_fast()
    xy_points = list()
    squares = list()
    circles = list()
    for idxs in x:
        (cent_x, cent_y) = (idxs[0], idxs[1])
        (width, height) = (idxs[2], idxs[3])
        left = cent_x - (width / 2)
        top = cent_y - (height / 2)
        try:
            rad = (width / 2) + (height ** 2 / (8 * width))
        except ZeroDivisionError:
            rad = 0

        # collect bounding box rectangle's value to generate rectangular polygon
        (x0, y0) = (left, top)
        (x1, y1) = (left + width, top)
        (x2, y2) = (left + width, top + height)
        (x3, y3) = (left, top + height)
        square = Polygon(((x0, y0), (x1, y1), (x2, y2), (x3, y3)))

        theta = np.linspace(0, 2 * np.pi, 8)
        x, y = rad * np.cos(theta) + cent_x, rad * np.sin(theta) + cent_y
        # build the list of points of circular geometry
        ext = list()
        for i_theta in range(len(theta)):
            ext.append((x[i_theta], y[i_theta]))
        # create center-point of boxes
        pts = Point((cent_x, cent_y))
        xy_points.append([pts])
        bulat = Polygon(ext)
        circles.append([bulat])
        squares.append([square])

    return xy_points, circles, squares


def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    elif weights.endswith('.weights'):
        load_darknet_weights(model, weights)
    else:  # darknet format
        load_darknet_weights(model, weights)

    # read main image
    source_meta = geo_params(source)

    # image = cv2.imread(source)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    image0 = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(image0.half() if half else image0.float()) if device.type != 'cpu' else None  # run once

    preds_list = list()
    preds_list0 = list()
    preds_conf = list()
    points_geom = list()
    detections = list()
    temp_dir = None

    for file0 in sliding_window(source, windowSize=(imgsz, imgsz)):
        (temp_dir, image0) = file0
        # get geographic parameters of input image, return error if failed!
        image_crs, affine_coord, src_meta = geo_params(image0)

        # read each image in temporary folder using opencv
        image0 = cv2.imread(image0)  # BGR
        (_H0, _W0) = image0.shape[:2]
        image0 = letterbox(image0, new_shape=imgsz)[0]
        image0 = image0[:, :, ::-1].transpose(2, 0, 1)
        image0 = np.ascontiguousarray(image0)

        image0 = torch.from_numpy(image0).to(device)
        image0 = image0.half() if half else image0.float()  # uint8 to fp16/32
        image0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image0.ndimension() == 3:
            image0 = image0.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(image0, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # print('pred:', pred.shape)
        # print(pred)
        # preds_list0.append(pred)
        pred_xywh = pred

        pred0 = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        # print('pred after nms0:\n', pred0, '\n')
        # boxes = [None] * len(pred)
        # for i, pred in enumerate(pred):
        #     xyxy = xywh2xyxy(pred[:, :4])
        #     y0 = torch.zeros_like(xyxy) if isinstance(xyxy, torch.Tensor) else np.zeros_like(xyxy)
        #     (y0[:, 0], y0[:, 1]) = affine_coord * (xyxy[:, 0], xyxy[:, 1])
        #     (y0[:, 2], y0[:, 3]) = affine_coord * (xyxy[:, 2], xyxy[:, 3])
        #     y1 = torch.zeros_like(pred) if isinstance(pred, torch.Tensor) else np.zeros_like(pred)
        #     (y1[:, 0], y1[:, 1]) = affine_coord * (pred[:, 0], pred[:, 1])
        #     y1[:, 2] = y0[:, 2] - y0[:, 0]
        #     y1[:, 3] = y0[:, 1] - y0[:, 3]
        #     y1[:, 4] = pred[:, 4]
        #     y1[:, 5] = pred[:, 5]
        #     boxes[i] = y1
        #     y1 = torch.cat((y1[:, :4], pred[:, 4].unsqueeze(1), pred[:, 5].float().unsqueeze(1)), 1)

        xyxy = xywh2xyxy(pred_xywh[:, :4])
        # y0 = torch.zeros_like(xyxy) if isinstance(xyxy, torch.Tensor) else np.zeros_like(xyxy)
        # (y0[:, 0], y0[:, 1]) = affine_coord * (xyxy[:, 0], xyxy[:, 1])
        # (y0[:, 2], y0[:, 3]) = affine_coord * (xyxy[:, 2], xyxy[:, 3])
        (xyxy[:, 0], xyxy[:, 1]) = affine_coord * (xyxy[:, 0], xyxy[:, 1])
        (xyxy[:, 2], xyxy[:, 3]) = affine_coord * (xyxy[:, 2], xyxy[:, 3])

        # y1 = torch.zeros_like(pred) if isinstance(pred, torch.Tensor) else np.zeros_like(pred)
        # (y1[:, 0], y1[:, 1]) = affine_coord * (pred[:, 0], pred[:, 1])
        # y1[:, 2] = y0[:, 2] - y0[:, 0]
        # y1[:, 3] = y0[:, 1] - y0[:, 3]
        # y1[:, 4] = pred[:, 4]
        # y1[:, 5] = pred[:, 5]
        (pred_xywh[:, 0], pred_xywh[:, 1]) = affine_coord * (pred_xywh[:, 0], pred_xywh[:, 1])
        pred_xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        pred_xywh[:, 3] = xyxy[:, 1] - xyxy[:, 3]
        pred_xywh[:, 4] = pred[:, 4]
        pred_xywh[:, 5] = pred[:, 5]
        # print('another pred', pred_xywh.shape)
        # print(pred_xywh)
        # y1 = torch.cat((pred_xywh[:, :4], pred[:, 4].unsqueeze(1), pred[:, 5].float().unsqueeze(1)), 1)
        print("-------------------------------------------------------------------------------")
        pred1 = non_max_suppression(pred_xywh, opt.conf_thres, opt.iou_thres,
                                    multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        for i, det in enumerate(pred1):
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    (c1, c2) = ((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]))
                    (left, top) = (c1[0], c1[1])
                    (width, height) = (c2[0] - left, c2[1] - top)

                    cx = left + width / 2
                    cy = top + height / 2
                    (geom_cx, geom_cy) = affine_coord * (cx, cy)
                    points = Point((geom_cx, geom_cy))
                    print(points)
                    points_geom.append([points])

    # preds_list0 = torch.cat(preds_list0, 1)
    # print(preds_list0)
    # print('----\n')
    # preds_list = torch.cat(preds_list, 1)
    # print(preds_list)
    # predictions = non_max_suppression(preds_list0, opt.conf_thres, opt.iou_thres,
    #                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
    # print('****\n')
    # print(predictions)

    # for i, det in enumerate(predictions):
    #     if det is not None and len(det):
    #         for *xyxy, conf, cls in det:
    #             (c1, c2) = ((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]))
    #             (left, top) = (c1[0], c1[1])
    #             (width, height) = (c2[0] - left, c2[1] - top)
    #
    #             cx = left + width / 2
    #             cy = top + height / 2
    #             points = Point((cx, cy))
    #             points_geom.append([points])
    #
    df = gpd.GeoDataFrame(points_geom, columns=['geometry'])
    df.crs = source_meta[0]
    df.to_file(os.path.join(out, 'detection_results.gpkg'), layer='points', driver='GPKG')

    shutil.rmtree(temp_dir)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
