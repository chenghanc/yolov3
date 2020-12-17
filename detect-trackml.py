import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

# Structure for point in cartesian plane
class point:
     
    def __init__(self):
         
        self.x = 0
        self.y = 0
  
# Constant integers for directions
RIGHT = 1
LEFT = -1
ZERO = 0
  
def directionOfPoint(A, B, P):
     
    global RIGHT, LEFT, ZERO
    
    # Two vectors (AB) and (AP) where P(x,y) is the query point
    # Subtracting co-ordinates of 
    # point A from B and P, to 
    # make A as origin
    B.x -= A.x
    B.y -= A.y
    P.x -= A.x
    P.y -= A.y
  
    # Determining cross Product
    cross_product = B.x * P.y - B.y * P.x
  
    # Return RIGHT if cross product is positive
    if (cross_product > 0):
        return RIGHT
         
    # Return LEFT if cross product is negative
    if (cross_product < 0):
        return LEFT
  
    # Return ZERO if cross product is zero
    return ZERO


def cpr(xA, yA, xB, yB, xM, yM):

    # Two vectors (AB) and (MB) where M(xM,yM) is the query point
    # Subtracting co-ordinate of point A from B
    vec1 = (xB - xA, yB - yA)

    # Subtracting co-ordinate of point M from B
    vec2 = (xB - xM, yB - yM)

    # Determining cross product
    crpr = vec1[0]*vec2[1] - vec1[1]*vec2[0]

    return crpr

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
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
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
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    alert={'alert': ''}

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

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

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #n = (det[:, -1] == 0).sum()  # person detections: Counting Person
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # set the boundary (arbitrary)
                        boundary1 = 1310   # 880 350
                        boundary  = 440    # 900 480
                        boundary4 = 870    # 1920 300
                        auxline   = 60
                        auxline1  = 40
                        auxline2  = 130
                        # define the yellow line beyond which the alert should be activated
                        line_thickness = 2
                        x1, y1 = boundary1, 0
                        x2, y2 = boundary4, boundary
                        cv2.line(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
                        # Auxiliary line: Left or Right
                        x3, y3 = x2, y2
                        x4, y4 = x2 - auxline1, y2 + auxline
                        cv2.line(im0, (x3, y3), (x4, y4), (0, 255, 255), thickness=line_thickness)
                        # Auxiliary line: Top
                        x5, y5 = x4, y4
                        x6, y6 = x4 - auxline2, 1080
                        cv2.line(im0, (x5, y5), (x6, y6), (255, 255, 0), thickness=line_thickness)
                        # Auxiliary line: Bottom
                        x7, y7 = x3 + 120, 0
                        x8, y8 = x4 - 400, 1080
                        #cv2.line(im0, (x7, y7), (x8, y8), (255, 255, 255), thickness=8)
                        #
                        label = '%s' % (names[int(cls)])
                        labelc = '%s %.2f' % (names[int(cls)], conf)
                        # convert boxes from [xmin, ymin, xmax, ymax] to [x, y, w, h]
                        xn = (xyxy[0] + xyxy[2]) / 2
                        yn = (xyxy[1] + xyxy[3]) / 2
                        wn = xyxy[2] - xyxy[0]
                        hn = xyxy[3] - xyxy[1]
                        # vector1
                        #v1 = (x2 - x1, y2 - y1)
                        # vector2
                        #v2 = (x2 - xn, y2 - yn)
                        # calculate cross product
                        cp1 = cpr(x1,y1,x2,y2,xn,yn)
                        cp2 = cpr(x3,y3,x4,y4,xn,yn)
                        cp3 = cpr(x5,y5,x6,y6,xn,yn)
                        acp = cpr(x7,y7,x8,y8,xn,yn)
                        #print(x6-x5,x8-x7)
                        # Counting Person
                        #cv2.putText(im0,"Counting Person = " + str(int(n)), (400,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                        if label != 'person' and label != 'backpack' and label != 'suitcase' and label != 'train':
                            continue
                        if label == 'backpack':
                            plot_one_box(xyxy, im0, label=label, color=(0,0,0))
                        elif label == 'suitcase':
                            plot_one_box(xyxy, im0, label=label, color=(0,0,0))
                        elif label == 'train':
                            plot_one_box(xyxy, im0, label=label, color=(0,0,0))
                        #if cp1 < 0 or cp2 < 0 or cp3 < 0:
                        elif (cp1 < 0 or cp2 < 0 or cp3 < 0) and (acp) > 0:
                            plot_one_box(xyxy, im0, label=label, color=(0,0,255)) # red   # point_n is on one side
                            cv2.putText(im0,"Alarm!!", (40,40),cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 3)
                            cv2.circle(im0,(int(xn),int(yn)),2,(0,0,255),-1)
                            alert={'alert':True}
                        else:
                            plot_one_box(xyxy, im0, label=label, color=(255,0,0)) # blue  # point_n is on the other side
                            alert ={'alert':False}
                        #print(alert)
                        if alert['alert']==True:
                            alarm = '\n\nAlarm!! Person cross the yellow line and an alarm notification should be activated!!\n\n'
                            print(alarm)

                        """
                        Start of:
                        Determine direction of query point P(xn, yn)
                        """  
                        A = point()
                        B = point()
                        P = point()

                        A.x = boundary4  #-30
                        A.y = boundary   #10  # A(-30, 10)
                        B.x = boundary1  #29
                        B.y = 0          #-15 # B(29, -15)
                        P.x = xn         #15
                        P.y = yn         #28  # P(15, 28)

                        direction = directionOfPoint(A, B, P)

                        if (direction == -1):
                            print("Left Direction")
                        elif (direction == 1):
                            print("Right Direction")
                        else:
                            print("Point is on the Line")
                        """
                        End of:
                        Determine direction of query point P(xn, yn)
                        """  

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
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

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

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
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
