
This repo contains Ultralytics inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Credit to Joseph Redmon for YOLO  https://pjreddie.com/darknet/yolo/.

## Inference

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8`

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt`  

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.pt`  


## mAP

- mAP@0.5 run at `--iou-thr 0.5`, mAP@0.5...0.95 run at `--iou-thr 0.7`

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt --img 640 --augment
```


