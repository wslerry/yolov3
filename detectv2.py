import argparse
from sys import platform
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


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
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # read main image
    image = cv2.imread(source)

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
    detections = list()
    temp_dir = None

    for (x, y, file0) in sliding_window(source, windowSize=(imgsz, imgsz)):
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

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        # convert all prediction boxes to geographical values
        # get xyxy, confidence, and class from prediction results
        for i, det in enumerate(pred):
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    # print(*xyxy, conf, cls)
                    """
                    x1y1
                    +------+
                    |      |
                    +------+
                            x2y2
                    """
                    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
                    (top, left) = (c1[0], c1[1])
                    (bottom, right) = (c2[0], c2[1])

                    # convert to geographic. IMPORTANT!
                    (x1, y1) = torch.mul(affine_coord * (top, left))
                    (x2, y2) = torch.mul(affine_coord * (bottom, right))
                    _xyxy = x1, y1, x2, y2
                    print(_xyxy)

                    # save all boxes params into list with geographical values
                    # preds_list.append([x1, y1, x2, y2, conf, cls])
                    # with open(os.path.join(out, 'test.txt'), 'a') as file:
                    #     file.write(('%2f ' * 6 + '\n') % (*_xyxy, conf, round(int(cls), 0)))

    # prediction = non_max_suppression(torch.cat(preds_list, 1), opt.conf_thres, opt.iou_thres,
    #                                  multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
    # detections.extend(prediction)

    # print(detections)
    # # # Run second NMS
    # predictions = non_max_suppression_fast(preds_list, opt.conf_thres, opt.iou_thres,
    #                                  multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
    # print(predictions)
    shutil.rmtree(temp_dir)
    # print("pred_list: \n", preds_list)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
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
