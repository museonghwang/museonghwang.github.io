---
layout: post
title: OpenCV로 그림 그리기
category: OpenCV
tag: OpenCV
---

[![Hits](https://hits.sh/museonghwang.github.io.svg?view=today-total&style=for-the-badge&label=Visitors&color=007ec6)](https://hits.sh/museonghwang.github.io/)

<br>

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 그림 그리기

이미지나 비디오에 그림을 그리는 방법을 알아봅니다. 객체나 얼굴을 인식해서 그 영역에 사각형을 그려서 표시하고 그 이름을 글씨로 표시하는 등의 용도로 자주 활용됩니다.

그리기 예제를 진행하기 위해서는 그림판 역할을 할 이미지가 하나 필요한데, blank_500.jpg라는 이름의 500 × 500 픽셀 크기의 아무것도 없는 완전히 하얀 이미지를 사용합니다. 이 이미지는 아래의 코드로 생성하기 바랍니다.

```py
import cv2
import numpy as np

img = np.full((500,500,3), 255, dtype=np.uint8)
cv2.imwrite('./img/blank_500.jpg', img)
```



## 1. 직선 그리기

이미지에 직선을 그리는 함수는 `cv2.line()` 입니다.

cv2.line(img, start, end, color I, thickness,

img: 그림 그릴 대상 이미지, NumPy 배열•

lineType]): 직선 그리기

• start: 선 시작 지점 좌표(x, y)

end: 선 끝 지점 좌표(x, y).

• color: 선 색상, (Blue, Green, Red), 0~255

• thickness=1: 선 두께

lineType: 선 그리기 형식

• cv2.LINE_4: 4 연결 선 알고리즘

• cv2.LINE_8: 8 연결 선 알고리즘

cv2.LINE_AA: 안티에일리어싱(antialiasing, 계단 현상 없는 선)

2장 기본 입출력

•

OpenCV

38


Project -

img 이미지에 start 지점에서 end 지점까지 선을 그립니다. color는 선의 색상을 표현하는 것으로 0~255 사이의 값 3개로 구성해서 표현합니다. 각 숫자는 파랑, 초록, 빨강(BGR) 순서이며, 이 색상을 섞어서 다양한 색상을 표현합니다. 일반적으로 웹에서 사용하는 RGB 순서와 반대라는 것이 특징입니다. thickness는 선의 두께를 픽셀 단위로 지시하는데, 생략하면 1픽셀이 적용됩니다. lineType은 선을 표현하는 방식을 나타내는 것으로 사선을 표현하거나 두꺼운 선의 끝을 표현할 때 픽셀에 따른 계단 현상을 최소화하기 위한 알고리즘을 선택합니다. cv2.LINE_으로 시작하는 3개의 상수를 선택할 수 있습니다. cv2.LINE_4와 cv2.LINE_8은 각각 브레젠햄(Bresenham) 알고리즘의 4연결, 8연결을 의미하고 cv2.LINE_AA는 가우시안 필터를 이용합니다.
