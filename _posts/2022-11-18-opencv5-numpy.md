---
layout: post
title: OpenCV를 위한 NumPy
category: OpenCV
tag: OpenCV
---

[![Hits](https://hits.sh/museonghwang.github.io.svg?view=today-total&style=for-the-badge&label=Visitors&color=007ec6)](https://hits.sh/museonghwang.github.io/)

<br>

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 1. 창 관리

한 개 이상의 이미지를 여러 창에 띄우거나 각 창에 키보드와 마우스 이벤트를 처리하려면 창을 관리하는 기능이 필요합니다. 다음은 OpenCV가 제공하는 창 관리 관련 API들을 요약한 것입니다.

* `cv2.namedWindow(title [, option])` : 이름을 갖는 창 열기
    * `title` : 창 이름, 제목 줄에 표시
    * `option` : 창 옵션, `cv2.WINDOW_` 로 시작.
        * `cv2․WINDOW_NORMAL` : 임의의 크기, 사용자 창 크기 조정 가능
        * `cv2.WINDOW_AUTOSIZE` : 이미지와 같은 크기, 창 크기 재조정 불가능
* `cv2.moveWindow(title, x, y)` : 창 위치 이동
    * `title` : 위치를 변경할 창의 이름
    * `x, y` : 이동할 창의 위치
* `cv2.resizeWindow(title, width, height)` : 창 크기 변경
    * `title` : 크기를 경할 이름
    * `width, height` : 크기를 변경할 창의 폭과 높이
* `cv2.destroyWindow(title)` : 창닫기
    * `title` : 닫을 대상 창 이름
* `cv2.destroyAllWindows()` : 열린 모든 창 닫기

다음 코드는 창 관리 함수를 이용하는 예제입니다.

```py
'''창 관리 API 활용하기'''
import cv2