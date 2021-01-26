---
title: plt&cv2 笔记
date: 2020-02-14 13:39:29
tags: [python, CV]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/02/14/lgWzFDvHry8bsdB.jpg
---

{% meting "27901061" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# matplotlib.pyplot
## Turn off X display
We may encounter the following issue if we are using a system without UI (e.g. Linux, Unix):
> QXcbConnection: Could not connect to display xxxxx
> Could not connect to any X display

```python
import matplotlib
matplotlib.use('Agg')
```

## Plot with annotation (Object Detection Task)
```python
import matplotlib.pyplot as ply

def plot(img, BBoxes, Scores, Classes):
	# - top_img (np.array) : input image
	# - BBoxes (list) : all bboxes, [x_lt, y_lt, x_br, y_br] (lt--lefttop, br--bottomright)
	# - Scores (list) : scores for all bboxes
	# - Classes (list) : prediction classes for  all bboxes
	h, w, c = img.shape
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(img)
	for i in range(len(BBoxes)):
		bbox = BBoxes[i]
		bbox[0] = max(bbox[0], 0)
		bbox[1] = max(bbox[1], 0)
		bbox[2] = min(bbox[2], h)
		bbox[3] = min(bbox[3], w)
		bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
		score = Scores[i]
		cls = Classes[i]
		# Draw rectangle
		ax.add_patch(
			plt.Rectangle((bbox[0], bbox[1]),
						  bbox[2] - bbox[0],
						  bbox[3] - bbox[1], fill=False,
						  edgecolor='red', linewidth=2)
			)
		# Put text
		ax.text(bbox[0], bbox[1] - 10,
				'{:s} {:.3f}'.format(cls, score),
				bbox=dict(facecolor='blue', alpha=0.5),
				fontsize=10, color='white')
	plt.gca().xaxis.set_major_locator(plt.NullLocator())  
	plt.gca().yaxis.set_major_locator(plt.NullLocator())  
	plt.tight_layout()
	plt.margins(0,0)
	plt.savefig('xxxx', dpi=200, bbox_inches='tight')
```
效果如图：
![](https://i.loli.net/2020/02/14/6blPkigdvajserD.png)

# cv2
## Resize by padding
```python
import cv2

def resize_by_padding(img, height, width):
    top, bottom, left, right = 0, 0, 0, 0
    h, w, c = img.shape
    longer = max(h, w)
    if(h < longer):
        dh = longer - h
        top = dh // 2
        bottom = dh - top
    elif(w < longer):
        dw = longer - w
        left = dw // 2     
        right = dw - left
    
    BLACK = [0, 0, 0]
    # 以 padding 的方式填充到正方形
    constant = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))
```

## BufferedReader <==> np.array
- 在调用 Azure Cognitive Service 中 Vision API 的时候，我们需要把输入的图片（np.array）转换为 API 可以接受的格式（BufferedReader）：
```python
import cv2
from io import BufferedReader, BytesIO

img = cv2.imread('xxx.jpg')
stream = BufferedReader(BytesIO(cv2.imencode('.jpg', legend)[1].tobytes()))
```

- 当我们自己写了一个 swagger API，上传图像后我们需要对图像进行处理，但是后端实际上接收到的是 FileStorage 格式，我们要将其转换为图像（np.array）才能进行处理
```python
"""
This is the OCR API
Upload the barplot and return the results
---
tags:
  - OCR API
parameters:
  - name: img
    in: formData
    description: Upload the barplot image
    required: true
    type: file
responses:
  500:
    description: Unexpected error
  200:
    description: The results of the barplot recognition

"""
import cv2
import numpy as np
from io import BufferedReader, BytesIO

# request为API接收的数据，此处为用户上传的图像，'img'为API中的参数名，见上文swagger API的定义
stream = request.files['img'] # werkzeug.datastructures.FileStorage
buf = BufferedReader(stream) # io.BufferedReader
nparr = np.frombuffer(buf.read(), dtype=np.uint8) # np.array
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # np.array
# 以上可以简化为
# img = cv2.imdecode(np.frombuffer(BufferedReader(stream).read(), dtype=np.uint8), cv2.IMREAD_COLOR)
```