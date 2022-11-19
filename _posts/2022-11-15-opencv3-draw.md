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

OpenCV를 이용한 대부분의 작업은 파일로 된 이미지를 읽어서 적절한 연산을 적용하고 그 결과를 화면에 표시하거나 다른 파일로 저장하는 것입니다. 여기서는 이미지 파일을 읽고 화면에 표시하고 저장하는 방법을 중점적으로 살펴보겠습니다.



## 1. 이미지 읽기

OpenCV를 사용해서 이미지를 읽고 화면에 표시하는 가장 간단한 코드는 아래와 같습니다.

```py
'''이미지 파일을 화면에 표시'''
import cv2

img_file = "./img/wonyoung.jpg" # 표시할 이미지 경로            ---①
img = cv2.imread(img_file)      # 이미지를 읽어서 img 변수에 할당 ---②
