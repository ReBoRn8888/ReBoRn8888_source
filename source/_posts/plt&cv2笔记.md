---
title: plt&cv2 笔记
date: 2020-02-14 13:39:29
tags: [python, CV]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/02/14/lgWzFDvHry8bsdB.jpg
---

{% aplayer 'Ezios Family' 'Jesper Kyd' 'http://music.163.com/song/media/outer/url?id=27901061.mp3' 'https://i.loli.net/2020/02/14/lgWzFDvHry8bsdB.jpg' autoplay %}

# matplotlib.pyplot 
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