
## People do not stand behind the yellow line on the train platform

Design an algorithm that is combined with **Yolov4** and check if people do not stand behind the yellow line and thus an alarm notification can be activated. The method proposed in this scenario can be easily generalized to **intrusion detection** as long as the area is well defined and CCTV is fixed. This repo adds new inference code `detect-track.py` for **Yolov4** in PyTorch. The code works on Linux, MacOS and Windows.

## Inference

```bash
python detect-track.py --names data/coco.names --cfg cfg/yolov4.cfg --weights yolov4.weights --img-size 608 --conf-thres 0.4 --iou-thres 0.6 --source CDS32-1300-1400cut3.mp4 --view-img
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8`

## Appendix: Modification

- #### Set the boundary of auxiliary lines (arbitrary)
- #### Define the auxiliary lines beyond which an alarm notification should be activated
- #### Define Cross Product of two vectors (AB) and (MB) where

    * A = A (xA, yA)
    * B = B (xB, yB)
    * M = M (xM, yM) is the query point

```python    
def cpr(xA, yA, xB, yB, xM, yM):

    # Two vectors (AB) and (MB) where M(xM,yM) is the query point
    # Subtracting co-ordinate of point A from B
    vec1 = (xB - xA, yB - yA)

    # Subtracting co-ordinate of point M from B
    vec2 = (xB - xM, yB - yM)

    # Determining cross product
    crpr = vec1[0]*vec2[1] - vec1[1]*vec2[0]

    return crpr
```

- #### Calculate cross product of two vectors

```python    
.
.
.
cp = cpr(x1,y1,x2,y2,xn,yn)

if cp < 0:
    print("Point (xn,yn) is on one side")
elif cp > 0:
    print("Point (xn,yn) is on the other side")
else:
    print("Point (xn,yn) is on the same line")
.
.
.
```

- #### Remarks: The generalization to multiple lines or polygons is straightforward, e.g. 

```python    
.
.
.
cp1 = cpr(x1,y1,x2,y2,xn,yn)
cp2 = cpr(x3,y3,x4,y4,xn,yn)
cp3 = cpr(x5,y5,x6,y6,xn,yn)
acp = cpr(x7,y7,x8,y8,xn,yn)

if (cp1 < 0 or cp2 < 0 or cp3 < 0) and (acp) > 0:
    print("Point (xn,yn) is on one side")
else:
    print("Point (xn,yn) is on the other side")
.
.
.
```

- #### Use the sign of the determinant of vectors (AB, AM), where M(X,Y) is the query point:

```
position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
```

  * It is  `0`: on the line 
  * It is `+1`: on one side
  * It is `-1`: on the other side

---

## Example
![](test-yellow.gif)

![](test-yellow-track1.gif)

![](test-yellow-track3.gif)



