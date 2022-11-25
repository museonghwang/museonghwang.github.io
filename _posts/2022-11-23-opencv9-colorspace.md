---
layout: post
title: OpenCV Image Processing 관심영역
category: OpenCV
tag: OpenCV
---

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 컬러 스페이스

**영상에 색상과 명암을 표현하는 방법들과 각각의 차이 그리고 활용 방법**에 대해 살펴보겠습니다.



## 1. 디지털 영상의 종류

디지털화된 **이미지는 픽셀(pixel, 화소)이라는 단위가 여러 개 모여서 그림을 표현**합니다. 하나의 픽셀을 어떻게 구성하느냐에 따라 이미지를 구분할 수 있습니다.

<br>

### 바이너리(binary, 이진) 이미지

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203908635-60f90864-26c4-44f2-b17e-fe2d9ff911db.png">
</p>

**한 개의 픽셀을 두 가지 값으로만 표현한 이미지**를 **바이너리(binary, 이진) 이미지**라고 합니다. 두 가지 값은 0과 1을 사용하기도 하고 0과 255를 사용하기도 합니다. 보통 0은 검은색, 1이나 255는 흰색을 표시해서 **말 그대로 흰색과 검은색만으로 그림을 그리는 흑백 이미지**입니다. 표현할 수 있는 값이 두 가지밖에 없어서 값으로는 명암을 표현할 수 없고, 점의 밀도로 명암을 표현할 수 있습니다.

**영상 작업에서**는 피사체의 색상과 명암 정보는 필요 없고 **오직 피사체의 모양 정보만 필요할 때 이런 이미지를 사용**합니다.

<br>

### 그레이 스케일 이미지

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203910336-2b0c5672-6b6e-4146-b9a0-91228ee12b01.png">
</p>

흔히 **흑백 사진이라고 하는 것**이 **그레이 스케일 이미지**입니다. 엄밀히 따지면, 흑백 이미지는 바로 앞서 설명한 바이너리 이미지를 말하는 것입니다.

**그레이 스케일 이미지는 한 개의 픽셀을 0~255의 값으로 표현**합니다. 픽셀 값의 크기로 명암을 표현하는데, 가장 작은 값인 0은 가장 어두운 검은색을 의미하고 값이 점점 커질수록 밝은 색을 의미하다가 255까지 가면 가장 밝은 흰색을 나타냅니다. 빛이 하나도 없는 O(영, zero)인 상태가 가장 어둡다고 생각하면 기억하기 쉽습니다. **한 픽셀이 가질 수 있는 값이 0~255이므로 음수가 없어서 부호 없는 1바이트의 크기로 표현하는 것이 일반적**입니다. **이미지 프로세싱에서는 색상 정보가 쓸모없을 때 컬러 이미지의 색상 정보를 제거함으로써 연산의 양을 줄이려고 그레이 스케일 이미지를 사용합니다.**

<br>

### 컬러 이미지

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203910345-27af2a83-7b9e-45ab-915c-0d359ce5b43e.png">
</p>

컬러 이미지에 색상을 표현하는 방법은 무척 다양합니다. 색상을 표현하는 방법에 따라 다르기는 하지만, 흔히 **컬러 이미지는 한 픽셀당 0~255의 값 3개를 조합해서 표현**합니다. **각 바이트마다 어떤 색상 표현의 역할을 맡을지를 결정하는 시스템**을 **컬러 스페이스(color space, 색공간)**라고 합니다. 컬러 스페이스의 종류는 **RGB, HSV, YUV(YCbCr), CMYK** 등 여러 가지가 있습니다.

<br>



## 2. RGB, BGR, RGBA

컴퓨터로 이미지에 색상을 표현하는 방법 중 가장 많이 사용하는 방법이 **RGB(Red,Green, Blue) 컬러 스페이스**입니다. **RGB는 빛의 3원소인 빨강, 초록, 파랑 세 가지색의 빛을 섞어서 원하는 색을 표현**합니다.

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203911151-b23cde2e-3605-4421-8385-558045d8d58a.png">
</p>

**각 색상은 0~255 범위로 표현하고 값이 커질수록 해당 색상의 빛이 밝아지는 원리**로 색상의 값이 모두 255일 때 흰색으로 표현되고, 모든 색상 값이 0일 때 검은색이 표현됩니다.

세 가지 색상을 표현하므로 RGB 이미지는 3차원 배열로 표현됩니다.

$$
row \times column \times channel
$$

**영상의 크기에 해당하는 행(row, height)과 열(column, width)에 세 가지 색상을 표현하는 차원이 추가**되는데, 이것을 **채널(channel)**이라고 합니다. 그러니까 RGB는 3개의 채널로 색상을 표현하는 컬러 스페이스인데, OpenCV는 그 순서를 반대로 해서 BGR 순서를 사용합니다.

**RGBA**는 **배경을 투명 처리하기 위해 알파(alpha) 채널을 추가한 것**을 말합니다. 4번째 채널의 값은 0~255로 표현할 수 있지만, 배경의 투명도를 표현하기 위해서는 0과 255만을 사용하는 경우가 많습니다.

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203913322-47bb8b07-a392-4d9a-962f-8a4e8e512a58.png">
</p>

`cv2.imread()` 함수의 두 번째 인자가 `cv2.IMREAD_COLOR` 인 경우 BGR로 읽어 들이고 `cv2.IMREAD_UNCHANGED` 인 경우 대상 이미지가 알파 채널을 가지고 있다면 BGRA로 읽어 들입니다. 다음 예제는 배경이 투명한 OpenCV 로고 이미지를 두 가지 옵션을 지정해서 비교합니다.

```py
'''BGR, BGRA, Ahlpha 채널'''
import cv2
import numpy as np

# 기본 값 옵션
img = cv2.imread('./img/opencv_logo.png')

# IMREAD_COLOR 옵션
bgr = cv2.imread('./img/opencv_logo.png', cv2.IMREAD_COLOR)

# IMREAD_UNCHANGED 옵션
bgra = cv2.imread('./img/opencv_logo.png', cv2.IMREAD_UNCHANGED)

# 각 옵션에 따른 이미지 shape
print("default", img.shape, "color", bgr.shape, "unchanged", bgra.shape)

cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:,:,3])  # 알파 채널만 표시
cv2.waitKey(0)
cv2.destroyAllWindows()

[output]
default (120, 98, 3) color (120, 98, 3) unchanged (120, 98, 4)
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203913446-809ecf17-6c17-4283-aa71-01c8b40cd42c.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203913470-74d25868-a680-4877-882f-8fcf0a2f8724.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/203913492-5b6daac7-b839-4cda-83e0-07b8c3449e34.png">
</p>

출력 내용을 보면 옵션을 따로 지정하지 않은 기본 옵션과 `cv2.IMREAD_COLOR` 옵션의 shape (240, 195, 3)로 동일한 것을 볼 수 있습니다. 위 두 그림은 투명한 배경이 검은색으로 표시되었고 로고 아래의 글씨도 검은색이다 보니 글씨가 보이지 않습니다. `cv2.IMREAD_UNCHANGED` 옵션으로 읽은 이미지는 shape가 (240, 195, 4)로 마지막 채널이 하나 더 있는 것을 알 수 있습니다. 이 채널만 떼어내서 따로 표시하였더니 로고와 글씨를 제외하고는 모두 검은색으로 표시됩니다. 즉, 전경은 255, 배경은 0의 값을 갖습니다. 이 **알파 채널의 정보를 이용하면 전경과 배경을 손쉽게 분리할 수 있어**서 **마스크 채널(mask channel)**이라고도 부릅니다.



## 3. 컬러 스페이스 변환

컬러 이미지를 그레이 스케일로 변환하는 것은 이미지 연산의 양을 줄여서 속도를 높이는 데 꼭 필요합니다. 이때 애초에 그레이 스케일로 읽어오는 방법은 앞서 2장에서 살펴본 cv2.imread(img, cv2.IMREAD_GRAYSCALE)입니다. 그런데 맨 처음에는 컬러 스케일로 읽어 들이고 필요에 따라 그레이 스케일이나 다른 컬러 스페이스로 변환해야 할 때도 많습니다.

그레이 스케일이나 다른 컬러 스페이스로 변환하는 방법은 변환 알고리즘을 직접 구현할 수도 있고, OpenCV에서 제공하는 cv2.cvtColor() 함수를 이용할 수도 있습니다.

아래의 [예제 4-6]은 컬러 스케일을 그레이 스케일로 변환하는 작업을 각각 보여줍니다.

이 예제에서 사용한 변환 알고리즘은 직접 구현하는 방법치고는 매우 쉬운 3채널의 평균 값을 구해서 그레이 스케일로 변환하는 방법입니다. 만약 변환 알고리즘이 매우 어렵다면 개발자에게는 큰 부담이 될 텐데 OpenCV를 사용하는 가장 큰 이유가 바로 이런 알고리즘을 정확히 몰라도 전체적인 원리만 알고 있으면 편리하게 작업할 수 있다는 것입니다.

[예제 4-6] BGR을 그레이 스케일로 변환(bgr2gray.py)

import cv2

import numpy as np

img = cv2.imread('../img/girl.jpg')

4.2 컬러 스페이스

117


OpenCV

img2

= img.astype(np. uint16)

b,g,r cv2.split(img2)

gray1 = ((b + g + r)/3).astype(np.uint8)

# dtype 변경 ---1

# 채널별로 분리 --- 2

# 평균값 연산 후 dtype 변경 ---3

gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR을 그레이 스케일로 변경 --- 4

cv2.imshow('original', img) cv2. imshow 'gray1', gray1) cv2.imshow('gray2', gray2)

cv2.waitKey(0) cv2.destroyAllWindows()

(x-316, y-0)-R:74 072849 (x-399, y-74)-184

[그림 4-10][예제 4-6]의 실행 결과

[예제 4-6] 에서 코드 ①, ②, ③은 평균 값을 구하는 알고리즘을 직접 구현했고, 코드 ④는 OpenCV에서 제공하는 함수를 이용한 방법입니다. 코드 ①에서 dtype을 uint16 타입으로 변경한 이유는 원래의 dtype이 uint8인 경우 평균 값을 구하는 과정에서 3채널의 값을 합하면 255보다 큰 값이 나올 수 있으므로 unit16으로 변경해서 계산을 마치고 다시 코드 ③에서 uint8로 변경합니다. 코드 ②에서 사용한 cv2.split() 함수는 매개변수로 전달한 이미지를 채널별로 분리해서 튜플로 반환합니다. 이 코드는 아래와 같은 NumPy 슬라이싱과 동일합니다.

b,g,r = img2[:,:, 0], img2[:,:,1], img2[:,:,2]

사실, 컬러 이미지를 그레이 스케일로 변환할 때 좀 더 정확한 명암을 얻으려면 단순히 평균 값만 계산하는 것보다 좀 더 정교한 연산이 필요합니다. 하지만, OpenCV에서 제공하는 cv2.cvtColor(img, flag) 함수는 이런 골치 아픈 알고리즘에서 우리를 자유롭게 해줍니다. 다음은 cv2.cvtColor() 함수에 대한 설명입니다.

4장 이미지 프로세싱 기초

118


flag: 변환할 컬러 스페이스, cv2.COLOR_로 시작하는 이름(274개)

Project

• out = cv2.cvtColor(img, flag)

.img: NumPy 배열, 변환할 이미지

cv2.COLOR_BGR2GRAY: BGR 컬러 이미지를 그레이 스케일로 변환

cv2.COLOR_GRAY2BGR: 그레이 스케일 이미지를 BGR 컬러 이미지로 변환

cv2.COLOR_BGR2RGB: BGR 컬러 이미지를 RGB 컬러 이미지로 변환

cv2.COLOR_BGR2HSV: BGR 컬러 이미지를 HSV 컬러 이미지로 변환

cv2.COLOR_HSV2BGR: HSV 컬러 이미지를 BGR 컬러 이미지로 변환

cv2.COLOR_BGR2YUV: BGR 컬러 이미지를 YUV 컬러 이미지로 변환cv2.COLOR_YUV2BGR: YUV 컬러 이미지를 BGR 컬러 이미지로 변환out: 변환한 결과 이미지(NumPy 배열)

컬러 스페이스 변환에 사용할 수 있는 플래그 상수는 2백여 개가 넘는데, 여기에 모두 다 기재하는 것은 의미가 없으므로 그들 중 중요하고 이 책에서 다루는 것들만 추려서 표시했습니다. 모든 상수는 이름이 cv2.COLOR_로 시작하므로 문서에서 쉽게 찾아볼 수 있을 것입니다. 파이썬 콘솔에서 아래의 코드를 실행해도 모든 플래그 상수를 출력해서 볼 수 있습니다.

>>> [i for i in dir(cv2) if i.startswith('COLOR_')}

[예제 4-6] 에서 코드 ④는 cv2.COLOR_BGR2GRAY 플래그 인자를 지정하는 것만으로도간단히 결과를 얻을 수 있습니다. 컬러 스케일 간의 변환은 컬러 스케일을 그레이 스케일로 바꾸는 것보다 좀 더 복잡한 알고리즘이 필요합니다. 하지만, 우리는 이 함수에 200여 가지의 플래그 인자를 지정하는 것만으로 컬러 스페이스 간의 변환을 쉽게처리할 수 있습니다.

cv2.COLOR_GRAY2BGR 플래그는 그레이 스케일을 BGR 스케일로 변환하는데, 실제로 흑백 사진을 컬러 사진으로 바꿔주는 것은 아닙니다. 2차원 배열 이미지를 3개 채널이 모두 같은 값을 갖는 3차원 배열로 변환하는 것입니다. 이 플래그는 영상 간에연산을 할 때 서로 차원이 다르면 연산을 할 수 없으므로 차원을 맞추는 용도로 주로사용합니다.

4.2 컬러 스페이스

.

.

.

.

.

119


4.2.4 HSV, HSI, HSL

HSV 포맷은 RGB와 마찬가지로 3채널로 컬러 이미지를 표시합니다. 3채널은 각각 H(Hue, 색조), S(Saturation, 채도), V(Value, 명도)입니다. 이때 명도를 표현하는 방법에 따라 마지막 V를 I(Intensity, 밀도)로 표기하는 HSI, 그리고 L(Lightness, 명도)로 표기하는 HSL 컬러 시스템도 있습니다. 이름에 차이가 있는 만큼 밝기 값을 계산하는 방법도 조금씩 차이가 있지만, 이 책에서는 편의상 같은 시스템으로 보고 HSV 기준으로 설명합니다.

HSV를 설명하는 데 가장 흔히 사용하는 방법은 다음 그림과 같은 원통형 시스템입니다.

[그림 4-11] HSV 컬러 스페이스

H 값은 그 픽셀이 어떤 색인지를 표현합니다. 원 위에 빨강에서 시작해서 노랑, 초록, 파랑을 거쳐 다시 빨강으로 돌아오는 방식으로 색상에 매칭되는 숫자를 매겨놓고 그 360° 범위의 값을 갖게 해서 색을 표현합니다. 하지만, OpenCV에서 영상을 표현할 때 사용하는 배열의 dtype은 최대 값이 255를 넘지 못하므로 360을 반으로 나누어 0~180 범위의 값으로 표현하고 180보다 큰 값인 경우에는 180으로 간주합니다.

90 75 105 60 120 45 30 150 15 165 135 0

[그림 4-12] 위에서 바라본 HSV 컬러 원통

4장 이미지 프로세싱 기초

- OpenCV

120


Project -

앞의 그림은 H 값만을 원통의 위에서 바라보는 시각으로 다시 그린 것에 각 색상별로 수치를 표시한 것입니다. 이 그림을 요약해서 대략 R, G, B 색상의 범위에 맞는 H값을 표시하면 아래와 같습니다.

빨강: 165~180, 0~15

초록: 45~75

파랑: 90~120

S 값은 채도, 포화도, 또는 순도로 해석할 수 있는데, 해당 색상이 얼마나 순수하게 포함되어 있는지를 표현합니다. S 값은 0~255 범위로 표현하며, 255는 가장 순수한 색상을 표현합니다.

V값은 명도로서 빛이 얼마나 밝은지 어두운지를 표현하는 값입니다. 이 값도 범위가 0~255이며, 255인 경우가 가장 밝은 상태이고 0(영, zero)인 경우가 가장 어두운 상태로 검은색이 표시됩니다.

BGR 포맷과 HSV 포맷 간의 변환은 cv2.cvtColor() 함수에 cv2.COLOR_BGR2HSV와 cv2.COLOR_HSV2BGR 플래그 상수를 이용합니다.

아래의 예제는 완전한 빨강, 초록, 파랑 그리고 노랑을 BGR 포맷으로 표현해서 HSV로 변환하여 어떤 값인지를 알아보는 예제입니다.

[예제 4-7] BGR에서 HSV로 변환(bgr2hsv.py)

import cv2

import numpy as np

# BGR 컬러 스페이스로 원색 픽셀 생성 ---① red_bgr = np.array([[[0,0,255]]], dtype=np.uint8) green_bgr = np.array([[[0,255,0]]], dtype=np.uint8) blue_bgr np.array([[[255,0,0]]], dtype=np.uint8) yellow_bgr = np.array([[[0,255,255]]], dtype=np.uint8) =

# 빨강 값만 갖는 픽셀

# 초록 값만 갖는 픽셀

파랑 값만 갖는 픽셀

# 노랑 값만 갖는 픽셀

# BGR 컬러 스페이스를 HSV 컬러 스페이스로 변환 ---2

red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV); green_hsv = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2HSV

blue_hsv = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2HSV);

yellow_hsv = cv2.cvtColor(yellow_bgr, cv2.COLOR_BGR2HSV);

#HSV로 변환한 픽셀 출력 print("red:",red_hsv) print("green:", green_hsv) print("blue", blue_hsv) print("yellow", yellow_hsv)

);

4.2 컬러 스페이스

●

121


앞의 예제 4기의 코드 ①에서 빨강, 초록, 파랑, 노랑에 해당하는 채널에만 최대 값인 255를 지정하고 나머지 채널에는 0(zero)을 지정해서 순도 높은 원색을 표현하고 코드 ②에서 HSV 컬러 스페이스로 변환한 후에 각 픽셀 값을 출력하고 있습니다. 출력 결과는 다음과 같습니다.

출력 결과 red: [[[ 0 255 255]]] green: [[[ 60 255 255]]] blue [[[120 255 255]]] yellow [[[ 30 255 255]]]

출력 결과를 살펴보면 가장 순도 높은 빨강의 H 값은 0, 초록은 60, 파랑은 120, 노랑은 30인 것을 확인할 수 있습니다.

픽셀의 색상이 궁금할 때 RGB 포맷의 경우 세 가지 채널의 값을 모두 조사해야 하지만, HSV 포맷은 오직 H 채널 값만 확인하면 되므로 색상을 기반으로 하는 여러 가지 작업에 효과적입니다.

4.2.5 YUV, YCbCr

YUV 포맷은 사람이 색상을 인식할 때 밝기에 더 민감하고 색상은 상대적으로 둔감한 점을 고려해서 만든 컬러 스페이스입니다. Y는 밝기(Luma)를 표현하고, U(Chroma Blue, Cb)는 밝기와 파란색과의 색상 차, V(Chroma Red, Cr)는 밝기와 빨간색과의 색상 차를 표현합니다. Y(밝기)에는 많은 비트수를 할당하고 U(Cb)와 V(Cr)에는 적은 비트 수를 할당해서 데이터를 압축하는 효과를 갖습니다.

Cb Cr -더--- 4:2:2 SA M LEE

[그림 4-13] YUV 개념도

4장 이미지 프로세싱 기초

OpenCV

122


Project

YUV라는 용어는 TV 방송에서 사용하는 아날로그 컬러 인코딩 시스템인 PAL(Phase Alternating Line)에서 정의한 용어입니다. YUV는 종종 YCbCr 포맷과 혼용되기도 하는데, 본래 YUV는 텔레비전 시스템에서 아날로그 컬러 정보를 인코딩하는 데 사용하고, YCbCr 포맷은 MPEG나 JPEG와 같은 디지털 컬러 정보를 인코딩하는 데 사용하였습니다. YUV는 요즘 들어 YCbCr로 인코딩된 파일 포맷을 설명하는 용어로 일반적으로 사용됩니다. 실제로도 YUV와 YCbCr은 RGB 포맷에서 변환하기 위한 공식이 달라서 OpenCV는 cv2.COLOR_BGR2YUV, cv2.COLOR_BGR2YCrCb가 따로 있고 변환 결과도 미세하게 다릅니다. 이 책에서는 편의상 같은 시스템으로 보고 YUV 컬러스페이스만 다릅니다. YUV는 밝기 정보와 컬러 정보를 분리해서 사용하므로 명암대비(contrast)가 좋지 않은 영상을 좋게 만드는 데 대표적으로 활용됩니다.

[예제 48]은 완전히 어두운 값과 완전히 밝은 값 그리고 중간 값을 BGR로 표현한 후에 YUV로 변환한 3개 채널을 살펴봅니다.

[예제 4-8] BGR에서 YUV로 변환(bgr2yuv.py)

import cv2

import numpy as np # BGR 컬러 스페이스로 세 가지 밝기의 픽셀 생성 ---1 dark = np.array([[[0,0,0]]], dtype=np.uint8) # 3채널 모두 0인 가장 어두운 픽셀 middle = np.array([[[127,127,127]]], dtype=np.uint8) # 3채널 모두 127인 중간 밝기 픽셀 bright = np.array([[[255,255,255]]], dtype=np.uint8)# 3채널 모두 255인 가장 밝은 픽셀 # BGR 컬러 스페이스를 YUV 컬러 스페이스로 변환 ---2 dark_yuv = cv2.cvtColor(dark, cv2.COLOR_BGR2YUV) middle_yuv = cv2.cvtColor(middle, cv2.COLOR_BGR2YUV) bright_yuv = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)

# YUV로 변환한 픽셀 출력 print("dark: ", dark_yuv) print("middle:", middle_yuv) print("bright", bright_yuv)

위 [예제 4-8]의 코드 ①에서 세 가지 밝기의 픽셀을 BGR 컬러 스페이스로 생성하고 나서 코드 ②에서 YUV 컬러 스페이스로 변환하고 출력합니다. 출력 결과는 다음과 같습니다.

출력 결과 dark: [[[ 0 128 128]]] middle: [[[127 128 128]]] bright [[[255 128 128]]]

4.2 컬러 스페이스

123


OpenCV

앞의 출력 결과에서 밝기 정도는 첫 번째 Y 채널에만 나타나는 것을 알 수 있습니다. 픽셀의 밝기를 제어해야 할 때 BGR 포맷은 3채널을 모두 연산해야 하지만, YUV 포맺은 Y채널 하나만 작업하면 되므로 효과적입니다.

