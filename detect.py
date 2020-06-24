import argparse
from sys import platform
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import rasterio as rio
# import geopandas as gpd
from shapely.geometry import Point, Polygon
from osgeo import ogr, osr
import tempfile

def detect(save_img=False):
    # img_size = (416, 256) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # lerryws version
    if not os.path.exists(out):
        os.mkdir(out)
    else:
        deleteFilesIn(out)

    if not os.path.exists(out + "/points"):
        os.makedirs(out + "/points", exist_ok=True)
    if not os.path.exists(out + "/canopy"):
        os.makedirs(out + "/canopy", exist_ok=True)

    # create temporary folder for tiles and as a new source folder
    os_temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(os_temp_dir, "yolov3_tiles_operation")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    else:
        deleteFilesIn(temp_path)

    # create tiles if image dimension is more then 2048
    # Get geographic data from image using rasterio
    with rio.open(source) as src:
        image_crs = src.crs
        id_crs = str(image_crs).split(":")
        id_crs = int(id_crs[1])
        src_meta = src.profile
        img_size = int(src_meta['width']/64) * 64
        if img_size > 1024:
            print("Image is bigger then 2048px, image will be tiled!")
            # new image size setting.
            img_size = 1024
            tile_img_size = img_size
            # then create tiles images
            to_tiles(opt.source, temp_path, tile_img_size, tile_img_size)
        else:
            pass

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        if img_size == 1024:
            dataset = LoadImages(temp_path, img_size=img_size)
        else:
            dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    total_predicted_box = list()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with rio.open(path) as src:
            affine = src.transform
            src_meta = src.profile

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # multipoint = ogr.Geometry(ogr.wkbMultiPoint)
        # create the spatial reference, WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(id_crs)

        # Create the output Driver
        outDriver = ogr.GetDriverByName('GeoJSON')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            file_path = os.path.dirname(save_path)
            filename = os.path.basename(save_path)
            filename = os.path.splitext(filename)[0]

            # Create the output GeoJSON
            pnt_path = os.path.join(os.path.join(file_path, "points"), filename + '.geojson')
            outDataSource = outDriver.CreateDataSource(pnt_path)
            outLayer = outDataSource.CreateLayer(filename, srs, geom_type=ogr.wkbPoint)

            # Create the output GeoJSON
            cpy_path = os.path.join(os.path.join(file_path, "canopy"), filename + '_canopy.geojson')
            outDataSource_canopy = outDriver.CreateDataSource(cpy_path)
            outLayer_canopy = outDataSource_canopy.CreateLayer(filename + '_canopy', srs, geom_type=ogr.wkbPolygon)

            ## create field attributes
            class_fld = ogr.FieldDefn("class", ogr.OFTString)
            outLayer.CreateField(class_fld)
            conf_fld = ogr.FieldDefn("confidences", ogr.OFTReal)
            outLayer.CreateField(conf_fld)
            x_fd = ogr.FieldDefn("x_easting", ogr.OFTReal)
            outLayer.CreateField(x_fd)
            y_fd = ogr.FieldDefn("y_northing", ogr.OFTReal)
            outLayer.CreateField(y_fd)

            ## create field attributes for canopy layer
            class_canopy_fld = ogr.FieldDefn("class", ogr.OFTString)
            outLayer_canopy.CreateField(class_canopy_fld)
            conf_canopy_fld = ogr.FieldDefn("confidences", ogr.OFTReal)
            outLayer_canopy.CreateField(conf_canopy_fld)
            rad_canopy_fld = ogr.FieldDefn("radius_m", ogr.OFTReal)
            outLayer_canopy.CreateField(rad_canopy_fld)

            # Get the output Layer's Feature Definition
            featureDefn = outLayer.GetLayerDefn()
            featureDefn_canopy = outLayer_canopy.GetLayerDefn()
            # create a new feature
            outFeature = ogr.Feature(featureDefn)
            outFeature_canopy = ogr.Feature(featureDefn_canopy)

            s += '%gx%g ' % img.shape[2:]  # print string

            # xy_coords = list()
            # circle_ = list()

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    total_predicted_box.append(n)

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=None, color=colors[int(cls)])

                    if opt.save_geom:
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        (left, top) = (c1[0], c1[1])
                        (width, height) = (c2[0] - left, c2[1] - top)

                        cent_x = left + (width / 2)
                        cent_y = top + (height / 2)

                        # try:
                        #     rad = (width / 2) + (height ** 2 / (8 * width))
                        # except ZeroDivisionError:
                        #     rad = 0

                        rad = (width / 2) + (height ** 2 / (8 * width))

                        xs, ys = affine * ([cent_x, cent_y])
                        point1 = ogr.Geometry(ogr.wkbPoint)
                        point1.AddPoint(xs, ys)
                        # multipoint.AddGeometry(point1)

                        outFeature.SetField("class", names[int(cls)])
                        outFeature.SetField("confidences", float(conf))
                        outFeature.SetField("x_easting", xs)
                        outFeature.SetField("y_northing", ys)

                        # Set new geometry
                        outFeature.SetGeometry(point1)

                        # Add new feature to output Layer
                        outLayer.CreateFeature(outFeature)

                        # xy_coords.append([names[int(cls)], float(conf), xs, ys, Point((xs, ys))])

                        theta = np.linspace(0, 2 * 3.14, 30)
                        x_, y_ = affine * (rad * np.cos(theta) + cent_x, rad * np.sin(theta) + cent_y)
                        radius = math.sqrt((x_[0] - xs) ** 2 + (y_[0] - ys) ** 2)

                        # ext = list()

                        # Create ring
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        # loop over x,y, add each point to list
                        for i_ in range(len(theta)):
                            ring.AddPoint(x_[i_], y_[i_])
                            # ext.append((x_[i_], y_[i_]))

                        poly = ogr.Geometry(ogr.wkbPolygon)
                        poly.AddGeometry(ring)

                        outFeature_canopy.SetField("class", names[int(cls)])
                        outFeature_canopy.SetField("confidences", float(conf))
                        outFeature_canopy.SetField("radius_m", round(radius, 2))

                        # Set new geometry
                        outFeature_canopy.SetGeometry(poly)

                        # Add new feature to output Layer
                        outLayer_canopy.CreateFeature(outFeature_canopy)

                        # circle_.append([names[int(cls)], float(conf), round(radius, 2), Polygon(ext)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if opt.save_geom:
                    frame2rasterio = np.transpose(im0, (2, 0, 1))

                    with rio.open(file_path + '/' + filename + '.tif', 'w', **src_meta) as dst:
                        dst.write(frame2rasterio, [3, 2, 1])
                else:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)

            if opt.save_geom:
                # dereference the feature
                outFeature = None
                outFeature_canopy = None
                # Save and close DataSources
                outDataSource = None
                outDataSource_canopy = None

                # srs.MorphToESRI()
                # file = open(file_path + '/' + filename + '.prj', 'w')
                # file.write(srs.ExportToWkt())
                # file.close()

                # df = gpd.GeoDataFrame(xy_coords, columns=['labels', 'confidences', 'x_easting', 'y_northing', 'geometry'])
                # df.crs = image_crs
                # df.to_file(save_path + '.geojson', driver='GeoJSON')
                #
                # df_circle = gpd.GeoDataFrame(circle_, columns=['labels', 'confidences', 'radius', 'geometry'])
                # df_circle.crs = image_crs
                # df_circle.to_file(save_path + '_circle.geojson', driver='GeoJSON')
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    vrt_filename = os.path.splitext(os.path.basename(source))[0]
    listOfFiles = tiles_list(out)
    vrt_output = out + "/" + str(vrt_filename) + ".vrt"
    vrt_opt = gdal.BuildVRTOptions(VRTNodata='none', srcNodata="NaN")
    gdal.BuildVRT(vrt_output, listOfFiles, options=vrt_opt)
    shutil.rmtree(temp_path)
    print(f'Total predicted {names[int(c)]} : {sum(total_predicted_box)} ')
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-geom', action='store_true', help='save results to *.shp')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
