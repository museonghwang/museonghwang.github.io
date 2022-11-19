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

* `cv2.line(img, start, end, color [, thickness, lineType])` : 직선 그리기
    * `img` : 그림 그릴 대상 이미지, NumPy 배열
    * `start` : 선 시작 지점 좌표(x, y)
    * `end` : 선 끝 지점 좌표(x, y)
    * `color` : 선 색상, (Blue, Green, Red), 0~255
    * `thickness=1` : 선 두께
    * `lineType` : 선 그리기 형식
        * `cv2.LINE_4` : 4 연결 선 알고리즘
        * `cv2.LINE_8` : 8 연결 선 알고리즘
        * `cv2.LINE_AA` : 안티에일리어싱(antialiasing, 계단 현상 없는 선)

`img` 이미지에 `start` 지점에서 `end` 지점까지 선을 그립니다. `color` 는 선의 색상을 표현하는 것으로 0~255 사이의 값 3개로 구성해서 표현합니다. 각 숫자는 파랑, 초록, 빨강(BGR) 순서이며, 이 색상을 섞어서 다양한 색상을 표현합니다. 일반적으로 웹에서 사용하는 RGB 순서와 반대라는 것이 특징입니다. `thickness` 는 선의 두께를 픽셀 단위로 지시하는데, 생략하면 1픽셀이 적용됩니다. `lineType` 은 선을 표현하는 방식을 나타내는 것으로 사선을 표현하거나 두꺼운 선의 끝을 표현할 때 픽셀에 따른 계단 현상을 최소화하기 위한 알고리즘을 선택합니다. `cv2.LINE_` 으로 시작하는 3개의 상수를 선택할 수 있습니다. `cv2.LINE_4` 와 `cv2.LINE_8` 은 각각 브레젠햄(Bresenham) 알고리즘의 4연결, 8연결을 의미하고 `cv2.LINE_AA` 는 가우시안 필터를 이용합니다.

다음 코드에서 다양한 선을 그려보면서 `cv2.line()` 함수의 매개변수의 의미를 알아봅니다.

```py
'''다양한 선 그리기'''
import cv2

img = cv2.imread('./img/blank_500.jpg')

# ---①
cv2.line(img, (50, 50), (150, 50), (255,0,0))   # 파란색 1픽셀 선
cv2.line(img, (200, 50), (300, 50), (0,255,0))  # 초록색 1픽셀 선
cv2.line(img, (350, 50), (450, 50), (0,0,255))  # 빨간색 1픽셀 선

# ---②
# 하늘색(파랑+초록) 10픽셀 선
cv2.line(img, (100, 100), (400, 100), (255,255,0), 10)
# 분홍(파랑+빨강) 10픽셀 선
cv2.line(img, (100, 150), (400, 150), (255,0,255), 10)
# 노랑(초록+빨강) 10픽셀 선
cv2.line(img, (100, 200), (400, 200), (0,255,255), 10)
# 회색(파랑+초록+빨강) 10픽셀 선
cv2.line(img, (100, 250), (400, 250), (200,200,200), 10)
# 검정 10픽셀 선
cv2.line(img, (100, 300), (400, 300), (0,0,0), 10)

# ---③
# 4연결 선
cv2.line(img, (100, 350), (400, 400), (0,0,255), 20, cv2.LINE_4)
# 8연결 선
cv2.line(img, (100, 400), (400, 450), (0,0,255), 20, cv2.LINE_8)
# 안티에일리어싱 선
cv2.line(img, (100, 450), (400, 500), (0,0,255), 20, cv2.LINE_AA)
# 이미지 전체에 대각선
cv2.line(img, (0, 0), (500, 500), (0,0,255))

cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202861744-95f304bd-1287-4df5-9e65-bd8fe4edaa2f.png">
</p>

사용한 이미지 blank_500.jpg는 흰색 배경에 아무 그림도 없는 텅빈 500 × 500 픽셀 크기의 이미지입니다. 코드 ①에서는 선 두께는 생략하여 두께가 1픽셀인 파란색, 초록색, 빨간색 선을 하나씩 그리고 있습니다. 코드 ②에서는 색상을 섞어서 다양한 색상의 10픽셀 선을 그리고 있습니다. 코드 ③은 사선이면서 두꺼운 선을 그리면서 계단 현상이 일어나는 것을 보여주고 있습니다. LINE_4와 LINE_8은 큰 차이를 느낄 수 없고, LINE_AA는 계단 현상을 없애는 것을 볼 수 있습니다.



## 2. 사각형 그리기

사각형을 그리는 함수는 `cv2.rectangle()` 입니다.

* `cv2.rectangle(img, start, end, color [, thickness, lineType])` : 사각형 그리기
    * `img` : 그림 그릴 대상 이미지, NumPy 배열
    * `start` : 사각형 시작 꼭짓점(x, y)
    * `end` : 사각형 끝 꼭짓점(x, y)
    * `color` : 색상(Blue, Green, Red)
    * `thickness`: 선 두께
        * `-1` : 채우기
    * `lineType` : 선 타입, `cv2.line()` 과 동일

사각형을 그릴 때 사용하는 `cv2.rectangle()` 함수는 앞서 설명한 `cv2.line()` 함수와 사용법이 거의 비슷합니다. 다만, 선이 아닌 면을 그리는 것이므로 선의 두께를 지시하는 `thickness` 에 `-1` 을 지정하면 사각형 면 전체를 color로 채우기를 합니다. 사각형을 그리기 위한 좌표는 시작 지점의 좌표 두 쌍과 그 반대 지점의 좌표 두 쌍으로 표현합니다.

```py
'''사각형 그리기'''
import cv2

img = cv2.imread('./img/blank_500.jpg')

# 좌상, 우하 좌표로 사각형 그리기
cv2.rectangle(img, (50, 50), (150, 150), (255,0,0))
# 우하, 좌상 좌표로 사각형 그리기
cv2.rectangle(img, (300, 300), (100, 100), (0,255,0), 10)
# 우상, 좌하 좌표로 사각형 채워 그리기 ---①
cv2.rectangle(img, (450, 200), (200, 450), (0,0,255), -1)

cv2.imshow('rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202862030-ed861ce4-a078-4c08-ab86-f305dbe74bdd.png">
</p>

보통 많은 그리기 도구에서 사각형을 그릴 때는 좌상단 꼭짓점과 우하단 꼭짓점 좌표를 사용하는 경우가 많은데, `cv2.rectangle()` 함수는 어느 지점이든 시작 지점과 그 반대 지점을 사용한다는 것이 특징입니다. 사각형의 크기는 두 좌표의 차이만큼이 됩니다. 코드 ①에서 선의 두께를 지정해야 하는 값에 -1을 전달해서 채우기로 그렸습니다.



## 3. 다각형 그리기

다각형을 그리는 함수는 `cv2.polylines()` 입니다.

* `cv2.polylines(img, points, isClosed, color [, thickness, lineType])` : 다각형 그리기
    * `img` : 그림 그릴 대상 이미지
    * `points` : 꼭짓점 좌표, NumPy 배열 리스트
    * `isClosed` : 닫힌 도형 여부, True/False
    * `color` : 색상(Blue, Green, Red)
    * `thickness` : 선 두께
    * `lineType` : 선 타입, `cv2.line()` 과 동일

이 함수의 `points` 인자는 다각형을 그리기 위한 여러 개의 꼭짓점 좌표를 전달합니다. 이때 좌표를 전달하는 형식이 지금까지와는 달리 NumPy 배열 형식입니다. `isClosed` 인자는 `Boolean` 타입인데, `True` 는 첫 꼭짓점과 마지막 꼭짓점을 연결해서 닫힌 도형(면)을 그리게 하고, `False` 는 단순히 여러 꼭짓점을 잇는 선을 그리게 합니다.

```py
'''다각형 그리기'''
import cv2
import numpy as np  # 좌표 표현을 위한 numpy 모듈 ---①

img = cv2.imread('./img/blank_500.jpg')

# Numpy array로 좌표 생성 ---②
# 번개 모양 선 좌표
pts1 = np.array([[50, 50], [150, 150], [100, 140], [200, 240]], dtype=np.int32)
# 삼각형 좌표
pts2 = np.array([[350, 50], [250, 200], [450, 200]], dtype=np.int32)
# 삼각형 좌표
pts3 = np.array([[150, 300], [50, 450], [250, 450]], dtype=np.int32)
# 5각형 좌표
pts4 = np.array([[350, 250], [450, 350], [400, 450], [300, 450], [250, 350]],\
                 dtype=np.int32) 

# 다각형 그리기 ---③
cv2.polylines(img, [pts1], False, (255,0,0))    # 번개 모양 선 그리기
cv2.polylines(img, [pts2], False, (0,0,0), 10)  # 3각형 열린 선 그리기 ---④
cv2.polylines(img, [pts3], True, (0,0,255), 10) # 3각형 닫힌 도형 그리기 ---⑤
cv2.polylines(img, [pts4], True, (0,0,0))       # 5각형 닫힌 도형 그리기

cv2.imshow('polyline', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202862189-b731a1ea-d3f8-4657-bc13-76512499436b.png">
</p>

코드 ①에서 NumPy 배열을 생성하기 위해 새로운 모듈을 임포트합니다. 코드 ②와 같이 그리기에 사용할 좌표들을 작성합니다. 실제로 다각형을 그리는 데 사용하는 함수는 코드 ③에서부터 나타납니다. 코드 ④와 ⑤에 사용한 좌표는 시작 위치만 다를 뿐 3개의 꼭짓점을 같은 비율로 표현하고 있는데, 열린 도형과 닫힌 도형의 차이를 지정하는 세 번째 인자의 차이로 각각 선과 도형으로 그려지는 것을 알 수 있습니다. `cv2.polylines()` 함수는 선의 굵기를 표현하는 인자에 -1을 지정해서 채우기 효과를 나타내는 것은 지원하지 않습니다.



## 4. 원, 타원, 호 그리기

원과 타원 그리고 호를 그리기 위한 함수는 다음과 같습니다.

* `cv2.circle(img, center, radius, color [, thickness, lineTypel)` : 원 그리기 함수
    * `img` : 그림 대상 이미지
    * `center` : 원점 좌표(x, y)
    * `radius` : 원의 반지름
    * `color` : 색상(Blue, Green, Red)
    * `thickness` : 선 두께(-1: 채우기)
    * `lineType` : 선 타입, `cv2.line()` 과 동일
* `cv2.ellipse(img, center, axes, angle, from, to, color [, thickness, lineType])` : 호나 타원 그리기 함수
    * `img` : 그림 대상 이미지
    * `center` : 원점 좌표(x, y)
    * `axes` : 기준 축 길이
    * `angle` : 기준 축 회전 각도
    * `from, to` : 호를 그릴 시작 각도와 끝 각도

완전한 동그라미를 그릴 때 가장 좋은 함수는 `cv2.circle()` 입니다. 하지만, 이 함수로는 동그라미의 일부분, 즉 호를 그리거나 찌그러진 동그라미인 타원을 그리는 것은 불가능하며, 이런 호나 타원을 그리려면 `cv2.ellipse()` 함수를 써야 합니다. 당연히 `cv2.ellipse()` 함수를 쓰는 것이 조금 더 어렵습니다.

다음 코드는 원과 타원 그리고 호를 그리는 방법을 보여주고 있습니다.

```py
'''원, 타원, 호 그리기'''
import cv2

img = cv2.imread('./img/blank_500.jpg')

# 원점(150,150), 반지름 100 ---①
cv2.circle(img, (150, 150), 100, (255,0,0))
# 원점(300,150), 반지름 70 ---②
cv2.circle(img, (300, 150), 70, (0,255,0), 5)
# 원점(400,150), 반지름 50, 채우기 ---③
cv2.circle(img, (400, 150), 50, (0,0,255), -1)

# 원점(50,300), 반지름(50), 회전 0, 0도 부터 360도 그리기 ---④
cv2.ellipse(img, (50, 300), (50, 50), 0, 0, 360, (0,0,255))
# 원점(150, 300), 아래 반원 그리기 ---⑤
cv2.ellipse(img, (150, 300), (50, 50), 0, 0, 180, (255,0,0))
#원점(200, 300), 윗 반원 그리기 ---⑥
cv2.ellipse(img, (200, 300), (50, 50), 0, 181, 360, (0,0,255))

# 원점(325, 300), 반지름(75,50) 납작한 타원 그리기 ---⑦
cv2.ellipse(img, (325, 300), (75, 50), 0, 0, 360, (0,255,0))
# 원점(450,300), 반지름(50,75) 홀쭉한 타원 그리기 ---⑧
cv2.ellipse(img, (450, 300), (50, 75), 0, 0, 360, (255,0,255))

# 원점(50, 425), 반지름(50,75), 회전 15도 ---⑨
cv2.ellipse(img, (50, 425), (50, 75), 15, 0, 360, (0,0,0))
# 원점(200,425), 반지름(50,75), 회전 45도 ---⑩
cv2.ellipse(img, (200, 425), (50, 75), 45, 0, 360, (0,0,0))

# 원점(350,425), 홀쭉한 타원 45도 회전 후 아랫 반원 그리기 ---⑪
cv2.ellipse(img, (350, 425), (50, 75), 45, 0, 180, (0,0,255))
# 원점(400,425), 홀쭉한 타원 45도 회전 후 윗 반원 그리기 ---⑫
cv2.ellipse(img, (400, 425), (50, 75), 45, 181, 360, (255,0,0))

cv2.imshow('circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202862480-ee3d8cee-7325-4631-a3da-24c4681ef057.png">
</p>

코드 ①, ②, ③은 `cv2․circle()` 함수를 이용해서 원을 그리고 있습니다. 주요 인자는 원점의 좌표와 반지름 값이므로 사용이 편리합니다. ③에서는 선의 두께 값에 `-1`을 전달하여 채우기 효과를 내고 있습니다.

나머지 코드는 모두 `cv2.ellipse()` 함수로 원, 타원, 호를 그리고 있습니다. 코드 ④처럼 이 함수로도 완전한 원을 그릴 수 있습니다. 반지름의 크기를 같은 비율로 지정하고, 회전 각도는 0으로, 표시할 호는 0도에서 360도를 모두 지정하였습니다. 코드 ⑤와 ⑥은 코드 ④와 똑같이 정확한 원을 표시하고 나서 표시할 호의 시작과 끝각을 0, 180 그리고 181, 360으로 원의 아랫부분과 윗부분에 해당하는 반원만 그렸습니다. 이렇게 호를 표시하고자 할 때 시작 각도는 3시 방향에서 시작하여 시계 방향으로 돌면서 6시 방향에서 90도, 9시 방향에서 180도와 같은 방식으로 3시 방향에서 360도까지 진행합니다.

코드 ⑦과 ⑧은 원의 반지름 값을 50과 75로 각각 다르게 지정해서 타원을 그립니다. 코드 ⑨와 ⑩은 타원을 15도와 45도만큼 회전하였습니다. 회전 각도는 0~360 사이의 각도를 지정하고, 필요에 따라 음수를 지정해서 회전 방향을 반대로 할 수도 있습니다.

코드 ⑪과 ⑫는 회전한 타원의 표시 각을 지정해서 타원의 아랫부분과 윗부분 호를 표시합니다. 회전한 원이나 타원에 대한 호를 표시할 때의 각도 값은 원래의 3시 방향에서 0도였던 것보다 회전한 각도만큼 더 이동해서 시작합니다.



## 5. 글씨 그리기

문자열을 이미지에 표시하는 함수는 `cv2.putText()` 입니다.

* `cv2.putText(img, text, point, fontFace, fontSize, color [, thickness, lineType])`
    * `img` : 글씨를 표시할 이미지
    * `text` : 표시할 문자열
    * `point` : 글씨를 표시할 좌표(좌측 하단 기준)(x, y)
    * `fontFace` : 글꼴
        * `cv2.FONT_HERSHEY_PLAIN` : 산세리프체 작은 글꼴
        * `cv2.FONT_HERSHEY_SIMPLEX` : 산세리프체 일반 글꼴
        * `cv2.FONT_HERSHEY_DUPLEX` : 산세리프체 진한 글꼴
        * `cv2.FONT_HERSHEY_COMPLEX_SMALL` : 세리프체 작은 글꼴
        * `cv2.FONT_HERSHEY_COMPLEX` : 세리프체 일반 글꼴
        * `cv2.FONT_HERSHEY_TRIPLEX` : 세리프체 진한 글꼴
        * `cv2.FONT_HERSHEY_SCRIPT_SIMPLEX` : 필기체 산세리프 글꼴
        * `cv2.FONT_HERSHEY_SCRIPT_COMPLEX` : 필기체 세리프 글꼴
        * `cv2.FONT_ITALIC` : 이탤릭체 플래그
    * `fontSize` : 글꼴 크기
    * `color`, `thickness`, `lineType` : `cv2.retangle()` 과 동일

point 좌표는 문자열의 좌측 하단을 기준으로 지정해야 합니다. 선택할 수 있는 글꼴의 종류는 위의 설명처럼 `cv2.FONT_HERSHEY_` 로 시작하는 상수로 정해져 있습니다. 크게 세리프(serif)체와 산세리프(sans-serif)체 그리고 필기체로 나뉘는데, 세리프체는 한글 글꼴의 명조체처럼 글자 끝에 장식을 붙여 모양을 낸 글꼴을 통틀어 말하며, 산세리프체는 고딕체처럼 획에 특별히 모양을 낸 것이 없는 글꼴을 말합니다. sans는 프랑스어로 '없다'는 뜻이고, serif는 타이포그래피에서 획의 끝이 돌출된 부분을 말하는 것으로 산세리프는 세리프가 없다는 뜻입니다.

OpenCV 상수에서는 상대적으로 단순한 모양인 산세리프체에 SIMPLEX라는 이름을 붙였고, 상대적으로 복잡한 모양인 세리프체에 COMLEX라는 이름을 붙인 것을 볼 수 있습니다.

```py
'''글씨 그리기'''
import cv2

img = cv2.imread('./img/blank_500.jpg')

# sans-serif small
cv2.putText(img, "Plain", (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
# sans-serif normal
cv2.putText(img, "Simplex", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
# sans-serif bold
cv2.putText(img, "Duplex", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
# sans-serif normall X2 ---①
cv2.putText(img, "Simplex", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,250))

# serif small
cv2.putText(img, "Complex Small", (50, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
# serif normal
cv2.putText(img, "Complex", (50, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
# serif bold
cv2.putText(img, "Triplex", (50, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0))
# serif normal X2 ---②
cv2.putText(img, "Complex", (200, 260), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))

# hand-wringing sans-serif
cv2.putText(img, "Script Simplex", (50, 330), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,0))
# hand-wringing serif
cv2.putText(img, "Script Complex", (50, 370), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,0,0))

# sans-serif + italic ---③
cv2.putText(img, "Plain Italic", (50, 430), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 1, (0,0,0))
# sarif + italic
cv2.putText(img, "Complex Italic", (50, 470), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (0,0,0))

cv2.imshow('draw text', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202862985-29daba49-938c-46bb-bb04-37361b3c7e41.png">
</p>

위 코드는 각각의 글꼴을 보여주고 있습니다. 코드 ①에서 산세리프체의 일반 글꼴 크기를 2배로 표시하고, 코드 ②에서 세리프체의 일반 글꼴 크기를 2배로 표시하고 있습니다. 코드 ③은 산세리프체와 세리프체를 이탤릭 플래그와 함께 사용하는 방법을 보여주고 있습니다.




