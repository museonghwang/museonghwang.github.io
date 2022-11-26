---
layout: post
title: OpenCV Image Processing 히스토그램
category: OpenCV
tag: OpenCV
---

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 히스토그램

**히스토그램(histogram)**은 뭐가 몇 개 있는지 개수를 세어 놓은 것을 그림으로 표시한 것을 말합니다. **히스토그램은 영상을 분석하는 데 도움이 많이 됩니다.**



## 1. 히스토그램 계산과 표시

영상 분야에서의 히스토그램은 전체 영상에서 픽셀 값이 1인 픽셀이 몇 개이고 2인 픽셀이 몇 개이고 하는 식으로 픽셀 값이 255인 픽셀이 몇 개인지까지 세는 것을 말합니다. 그렇게 하는 이유는 전체 영상에서 픽셀들의 색상이나 명암의 분포를 파악하기 위해서입니다.

OpenCV는 영상에서 히스토그램을 계산하는 `cv2.calcHist()` 함수를 제공합니다.

* `cv2.calcHist(img, channel, mask, histSize, ranges)`
    * `img` : 입력 영상, [img]처럼 리스트로 감싸서 표현
    * `channel` : 처리할 채널, 리스트로 감싸서 표현
        * 1채널: [0], 2채널: [0, 1], 3채널: [0, 1, 2]
    * `mask` : 마스크에 지정한 픽셀만 히스토그램 계산
    * `histSize` : 계급(bin)의 개수, 채널 개수에 맞게 리스트로 표현
        * 1채널: [256], 2채널: [256, 256], 3채널: [256, 256, 256]
    * `ranges` : 각 픽셀이 가질 수 있는 값의 범위, RGB인 경우 [0, 256]

가장 간단하게 그레이 스케일 이미지의 히스토그램을 계산해서 그려보겠습니다.

```py
'''그레이 스케일 1채널 히스토그램'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 이미지 그레이 스케일로 읽기 및 출력
img = cv2.imread('./img/mountain.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)

#--② 히스토그램 계산 및 그리기
hist = cv2.calcHist([img], [0], None, [256], [0,255])
plt.plot(hist)

print("hist.shape:", hist.shape)  #--③ 히스토그램의 shape (256,1)
print("hist.sum():", hist.sum(), "img.shape:",img.shape) #--④ 히스토그램 총 합계와 이미지의 크기
plt.show()

[output]
hist.shape: (256, 1)
hist.sum(): 270000.0 img.shape: (450, 600)
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204102869-2ae10ab5-7962-4815-baef-b5d2604179e2.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204102875-1f3994e2-b77d-4d97-a013-a3b934cdf278.png">
</p>

위 코드는 영상을 그레이 스케일로 읽어서 1차원 히스토그램으로 출력하는 예제입니다. 코드 ②가 가장 핵심적인 코드입니다. 여기서 `cv2.calcHist()` 함수 호출에 사용한 인자를 순서대로 설명하면, 히스토그램 대상 이미지는 [img], 1채널만 있어서 [0], 마스크는 사용하지 않으므로 None, 가로축(x축)에 표시할 계급(bin)의 개수는 [256], 픽셀 값 중 최소 값과 최대 값은 [0, 256]이라는 의미입니다. 여기서 최대값은 범위에 포함되지 않으므로 255보다 1 큰 값을 전달합니다. 이렇게 얻은 결과를 `plt.plot()` 함수에 전달하면 히스토그램을 그림으로 보여줍니다.

코드 ③에서 출력한 히스토그램 배열의 shape는 (256, 1)입니다. 256개의 계급에 각각 픽셀 수가 몇 개인지 저장한 모양새입니다. 코드 ④에서는 히스토그램의 전체 합과 이미지의 크기를 출력하고 있는데, 이 값으로 이미지의 폭과 높이의 곱과 히스토그램의 합(450×600 = 270,000)이 같은 것을 알 수 있습니다.

그레이 스케일이 아닌 컬러 스케일에 대한 히스토그램은 3개 채널, 즉 R, G, B를 각각 따로 계산해서 그려볼 수 있습니다.

```py
'''컬러 히스토그램'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 이미지 읽기 및 출력
img = cv2.imread('./img/mountain.jpg')
cv2.imshow('img', img)

#--② 히스토그램 계산 및 그리기
channels = cv2.split(img)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 255])
    plt.plot(hist, color = color)
    
plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204103075-75b1c27f-93d2-4646-a1b4-5b773c6219b6.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204103079-bf5dedac-5649-407e-9f42-61f3307127f1.png">
</p>

위 예제는 컬러 스케일 이미지의 3개 채널에 대해서 1차원 히스토그램을 각각 구하고 나서 하나의 플롯에 그렸습니다. 히스토그램을 보면 파란 하늘이 가장 넓은 영역을 차지하고 있으므로 파란색 분포가 크고 초록 나무와 단풍 때문에 초록색과 빨간색의 분포가 그 뒤를 따르는 것으로 보입니다.

<br>



## 2. 노멀라이즈

**노멀라이즈(normalize, 정규화)는 원래 기준이 서로 다른 값을 같은 기준이 되게 만드는 것**을 말합니다. 노멀라이즈는 **서로 다른 기준을 하나의 절대적인 기준으로 만들기도 하지만 절대적인 기준 대신 특정 구간으로 노멀라이즈하면 특정 부분에 몰려 있는 값을 전체 영역으로 골고루 분포하게 할 수도 있습니다.** 예를 들어 전교생이 5명인 학생들의 성적이 95, 96, 97, 98, 99, 100점일 때 95점 이상에게 A+ 학점을 준다면 전교생이 A+를 받게 되니 이 시험엔 분명 문제가 있습니다. 선생님이 각 학생의 점수를 70 ~ 100점 사이로 다시 환산하고 싶어 한다면, 이때 필요한 것이 바로 **구간 노멀라이즈**입니다.

원래 점수는 95 ~ 100, 즉 5점 간격이었는데, 새로운 점수는 70 ~ 100, 즉 30점 간격이므로 $30 / 5 = 6$ 으로, 학생들의 성적이 70, 76, 82, 88, 94, 100점으로, 원래 점수 1점 차이는 새로운 점수 6점 차이가 됩니다. 원래 점수가 5점 구간에서 얼마인지 찾아 그 비율(6점)과 곱해서 새로운 시작 구간(70점)에 더하면 새로운 점수를 구할 수 있습니다. 이것을 수학식으로 정리하면 다음과 같습니다.

$$
I_N = (I - Min)\frac{newMax - newMin}{Max - Min} + newMin
$$

* $I$ : 노멀라이즈 이전 값
* $Min, Max$ : 노멀라이즈 이전 범위의 최소 값, 최대 값
* $newMin, newMax$ : 노멀라이즈 이후 범위의 최소 값, 최대 값
* $I_N$ : 노멀라이즈 이후 값

**영상 분야에서는 노멀라이즈를 가지고 픽셀 값들이 0~255에 골고루 분포하지 않고 특정 영역에 몰려 있는 경우 화질을 개선하기도 하고 영상 간의 연산을 해야 하는데, 서로 조건이 다른 경우 같은 조건으로 만들기도 합니다.**

OpenCV는 노멀라이즈 기능을 아래와 같은 함수로 제공합니다.

* `dst = cv2.normalize(src, dst, alpha, beta, type_flag)`
    * `src` : 노멀라이즈 이전 데이터
    * `dst` : 노멀라이즈 이후 데이터
    * `alpha` : 노멀라이즈 구간 1
    * `beta` : 노멀라이즈 구간 2, 구간 노멀라이즈가 아닌 경우 사용 안함
    * `type_flag` : 알고리즘 선택 플래그 상수
        * `cv2.NORM_MINMAX` : alpha와 beta 구간으로 노멀라이즈
        * `cv2.NORM_L1` : 전체 합으로 나누기, alpha = 노멀라이즈 전체 합
        * `cv2.NORM_L2` : 단위 벡터(unit vector)로 노멀라이즈
        * `cv2.NORM_INF` : 최대 값으로 나누기

아래의 예제는 뿌연 영상에 노멀라이즈를 적용해서 화질을 개선하는 예제입니다.

```py
'''히스토그램 정규화'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 그레이 스케일로 영상 읽기
img = cv2.imread('./img/abnormal.jpg', cv2.IMREAD_GRAYSCALE)

#--② 직접 연산한 정규화
img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

#--③ OpenCV API를 이용한 정규화
img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#--④ 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

cv2.imshow('Before', img)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204103697-4344e3c2-b0eb-4a0c-86e0-8edb022b67b7.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204103700-a4f30d85-524e-4dd5-9df8-e2a7aee4c091.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204103704-f968ec32-1323-4f35-b19d-a525e8c38aa2.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204103821-0d6f0c07-5416-46c5-8775-ebcaed60fd13.png">
</p>

위 코드 ②는 앞서 설명한 노멀라이즈 공식을 직접 대입해서 연산하였습니다. 코드 ②에서 dtype을 float32로 바꾸었다가 다시 uint8로 바꾼 이유는 연산 과정에서 소수점이 발생하기 때문입니다. 코드 ③은 `cv2.normalize()` 함수로 노멀라이즈를 적용했습니다. 이때 앞서 설명한 구간 노멀라이즈를 사용하려면 `cv2.NORM_MINMAX` 플래그 상수를 사용하고 alpha, beta는 대상 구간 값을 전달합니다. 실행 결과는 중앙에 몰려 있던 픽셀들의 분포가 전체적으로 고르게 펴져서 화질이 개선된 것을 보여줍니다.

구간 노멀라이즈가 아니라 서로 다른 히스토그램의 빈도를 같은 조건으로 비교하는 경우에는 전체의 비율로 노멀라이즈해야 하는데, 이때 코드는 다음과 같습니다.

```py
norm = cv2.normalize(hist, None, 1, 0, cv2.NORM_L1)
```

위 코드에서 `cv2.NORM_L1` 플래그 상수를 사용하면 결과는 전체를 모두 합했을 때 1이 됩니다. 세 번째 인자 값에 따라 그 합은 달라지고 네 번째 인자는 무시됩니다.

<br>



## 3. 이퀄라이즈

앞서 설명한 **노멀라이즈는 분포가 한곳에 집중되어 있는 경우에는 효과적이지만 그 집중된 영역에서 멀리 떨어진 값이 있을 경우에는 효과가 없습니다.** 다시 학생들 점수를 예로 들면 전교생 5명의 점수가 70, 96, 98, 98, 100으로 나왔다면 첫 번째 학생의 점수가 70점이므로 구간 노멀라이즈로는 새로운 70 ~ 100 분포로 만들어도 결과는 동일한데, 기존의 범위와 새로운 범위가 같기 때문입니다. 이때에는 **이퀄라이즈(equalize, 평탄화)**가 필요합니다.

**이퀄라이즈는 히스토그램으로 빈도를 구해서 그것을 노멀라이즈한 후 누적값을 전체 개수로 나누어 나온 결과 값을 히스토그램 원래 픽셀 값에 매핑합니다.** 히스토그램 이퀄라이즈를 위한 수학식은 아래와 같습니다.

$$
H'(v) = round\left(\frac{cdf(v) - cdf_{min}}{(M \times N) - cdf_{min}} \times (L - 1)\right)
$$

* $cdf(v)$ : 히스토그램 누적 함수
* $cdf_{min}$ : 누적 최소 값, 1
* $M × N$ : 픽셀 수, 폭 × 높이
* $L$ : 분포 영역, 256
* $round(v)$ : 반올림
* $H'(v)$ : 이퀄라이즈된 히스토그램 값

**이퀄라이즈는 각각의 값이 전체 분포에 차지하는 비중에 따라 분포를 재분배하므로 명암 대비(contrast)를 개선하는 데 효과적입니다.**

OpenCV에서 제공하는 이퀄라이즈 함수는 아래와 같습니다.

* `dst = cv2.equalizeHist(src[, dst])`
    * `src` : 대상 이미지, 8비트 1채널
    * `dst` : 결과 이미지

다음 예제는 어둡게 나온 사진을 그레이 스케일로 바꾸어 이퀄라이즈를 적용해서 개선시키는 예제입니다.

```py
'''그레이 스케일 이퀄라이즈 적용'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 대상 영상으로 그레이 스케일로 읽기
img = cv2.imread('./img/yate.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape[:2]

#--② 이퀄라이즈 연산을 직접 적용
hist = cv2.calcHist([img], [0], None, [256], [0, 256]) # 히스토그램 계산
cdf = hist.cumsum()                                    # 누적 히스토그램 
cdf_m = np.ma.masked_equal(cdf, 0)                     # 0(zero)인 값을 NaN으로 제거
cdf_m = (cdf_m - cdf_m.min()) / (rows * cols) * 255    # 이퀄라이즈 히스토그램 계산
cdf = np.ma.filled(cdf_m,0).astype('uint8')            # NaN을 다시 0으로 환원
print(cdf.shape)
img2 = cdf[img]                                        # 히스토그램을 픽셀로 맵핑

#--③ OpenCV API로 이퀄라이즈 히스토그램 적용
img3 = cv2.equalizeHist(img)

#--④ 이퀄라이즈 결과 히스토그램 계산
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

#--⑤ 결과 출력
cv2.imshow('Before', img)
cv2.imshow('Manual', img2)
cv2.imshow('cv2.equalizeHist()', img3)
hists = {'Before':hist, 'Manual':hist2, 'cv2.equalizeHist()':hist3}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204104480-e4d1a9e9-4c79-4abd-a11c-b399529a7fc8.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204104484-f3afc247-0cc8-465f-af04-5c9dd64c8c70.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204104487-bafc7578-04f2-4a2b-bb06-fd81d2a13c70.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204104471-98a94842-904c-44c6-8afd-c0d439041fe9.png">
</p>

코드 ②는 히스토그램 이퀄라이즈 수식을 그대로 연산에 적용하고 있습니다. `hist.cumsum()` 은 누적합을 구하는 함수이고, `np.ma.masked_equal(cdf, 0)` 은 요소 값이 0인 것을 NaN으로 적용하는데, 불필요한 연산을 줄이고자 하는 이유입니다. 이것을 다시 원래대로 되돌리는 기능이 `np.ma.filled(cdf_m, 0)` 입니다. `img2 = cdf[img]` 는 연산 결과를 원래의 픽셀 값에 매핑합니다.

이렇게 복잡한 연산에 OpenCV에서 제공하는 API를 사용하면 코드 ③처럼 단 한줄이면 끝납니다. 실행 결과를 보면 직접 계산을 적용한 결과와 `cv2.equalizeHist()` 함수를 사용한 것 모두 밝기가 개선된 것을 알 수 있습니다.

히스토그램 이퀄라이즈는 컬러 스케일에도 적용할 수 있는데, **밝기 값을 개선하기 위해서는 3개 채널 모두를 개선해야 하는 BGR 컬러 스페이스보다는 YUV나 HSV로 변환해서 밝기 채널만을 연산해서 최종 이미지에 적용하는 것이 좋습니다.**

다음 예제는 YUV 컬러 스페이스로 변경한 컬러 이미지에 대한 이퀄라이즈를 보여줍니다.

```py
'''컬러 이미지에 대한 이퀄라이즈 적용'''
import numpy as np, cv2

img = cv2.imread('./img/yate.jpg') #이미지 읽기, BGR 스케일

#--① 컬러 스케일을 BGR에서 YUV로 변경
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 

#--② YUV 컬러 스케일의 첫번째 채널에 대해서 이퀄라이즈 적용
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 

#--③ 컬러 스케일을 YUV에서 BGR로 변경
img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

cv2.imshow('Before', img)
cv2.imshow('After', img2)
cv2.waitKey()
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204105614-98292ab4-26cb-4cf8-8937-e66102230005.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204105612-9c364c5c-4d14-4a95-9374-0b100bc5e8b9.png">
</p>

요트 부분을 비교해서 보면 훨씬 선명한 결과를 얻은 것을 볼 수 있습니다. HSV의 세 번째 채널에 대해서 이퀄라이즈를 적용해도 비슷한 결과를 얻을 수 있습니다. 코드 ②를 HSV 컬러 스페이스에 적용하면 코드는 아래와 같습니다.

```py
img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
```

<br>



## 4. CLAHE

**CLAHE(Contrast Limiting Adaptive Histogram Equalization)는 영상 전체에 이퀄라이즈를 적용했을 때 너무 밝은 부분이 날아가는 현상을 막기 위해 영상을 일정한 영역으로 나눠서 이퀄라이즈를 적용하는 것**을 말합니다. 노이즈가 증폭되는 것을 막기 위해 어느 히스토그램 계급(bin)이든 지정된 제한 값을 넘으면 그 픽셀은 다른 계급으로 배분하고 나서 이퀄라이즈를 적용합니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204105767-fb2cf4b2-90a3-44f0-8127-0baae12fa85f.png">
</p>

CLAHE를 위한 OpenCV 함수는 다음과 같습니다.

* `clahe = cv2.createCLAHE(clipLimit, tileGridSize)` : CLAHE 생성
    * `clipLimit` : Contrast 제한 경계 값, 기본 40.0
    * `tileGridSize` : 영역 크기, 기본 8 × 8
    * `clahe` : 생성된 CLAHE 객체
* `clahe.apply(src)` : CLAHE 적용
    * `src` : 입력 영상

```py
'''CLAHE'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--①이미지 읽어서 YUV 컬러스페이스로 변경
img = cv2.imread('./img/bright.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

#--② 밝기 채널에 대해서 이퀄라이즈 적용
img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

#--③ 밝기 채널에 대해서 CLAHE 적용
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

#--④ 결과 출력
cv2.imshow('Before', img)
cv2.imshow('CLAHE', img_clahe)
cv2.imshow('equalizeHist', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204105914-68b6763c-c17c-4a3e-b3ae-02f3e6ce5a3c.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204105919-61e46985-bdaf-4640-bdfd-33a76c8390a1.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204105928-1a36aa68-4020-47d2-84d7-d939583ac9f7.png">
</p>

코드 ②는 단순한 이퀄라이즈를 적용했고 코드 ③은 CLAHE를 적용했습니다. `cv2.createCLAHE()` 에서 `clipLimit=3.0` 은 기본 값이 40.0이므로 상황에 따라적절한 값으로 바꾸어야 합니다. 원본 사진은 사진을 찍을 때 빛이 너무 많이 들어 갔습니다. 이퀄라이즈를 적용한 결과는 밝은 곳이 날아가는 증상이 발생한 것을 보여주고 있습니다.

<br>



### 5. 2D 히스토그램

1차원 히스토그램은 각 픽셀이 몇 개씩인지 세어서 그래프로 표현하는데, 2차원 히스토그램은 이와 같은 축이 2개이고 각각의 축이 만나는 지점의 개수를 표현합니다. 그래서 이것을 적절히 표현하려면 지금까지 사용한 2차원 그래프가 아닌 3차원 그래프가 필요합니다. 아래의 예제는 다음 그림의 맑고 화창한 가을 하늘의 산을 찍은 사진을 2차원 히스토그램으로 표현한 것입니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204106096-7bcea2a6-fce7-46fb-a93e-8a36ec03344e.jpg">
</p>

```py
'''2D 히스토그램'''
import cv2
import matplotlib.pylab as plt

plt.style.use('classic')        # --① 컬러 스타일을 1.x 스타일로 사용
img = cv2.imread('./img/mountain.jpg')

plt.subplot(131)
hist = cv2.calcHist([img], [0,1], None, [32,32], [0,256,0,256]) #--②
p = plt.imshow(hist)                                            #--③
plt.title('Blue and Green')                                     #--④
plt.colorbar(p)                                                 #--⑤


plt.subplot(132)
hist = cv2.calcHist([img], [1,2], None, [32,32], [0,256,0,256]) #--⑥
p = plt.imshow(hist)
plt.title('Green and Red')
plt.colorbar(p)

plt.subplot(133)
hist = cv2.calcHist([img], [0,2], None, [32,32], [0,256,0,256]) #--⑦
p = plt.imshow(hist)
plt.title('Blue and Red')
plt.colorbar(p)

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204106163-662b505a-535b-452d-8dff-04a4ec666c56.png">
</p>

코드 ②, ⑥, ⑦은 각각 파랑과 초록, 초록과 빨강, 파랑과 빨강에 대한 2차원 히스토그램을 계산합니다. 계급 수는 256으로 조밀하게 하면 색상이 너무 작게 표현되서 32 정도로 큼직하게 잡았습니다. 각 값의 범위는 0~256 이 두 번 반복됩니다. 계산한 히스토그램을 코드 ③에서 `imshow()` 함수로 표현했습니다. 그래서 이 결과를 보면서 정확한 정보를 읽는 것은 그다지 도움이 되지는 않습니다. 다만, 코드⑤에서 각 색상에 대한 컬러 막대를 범례(legend)로 표시했기 때문에 색상을 보면서 대략의 정보를 알아낼 수 있습니다. 빨간색으로 표시될수록 픽셀의 개수가 많고 파란색은 픽셀이 적은 것을 나타냅니다.

**여기서 중요한 것은 2차원 히스토그램의 의미**입니다. 왼쪽 그림은 파랑과 초록의 2차원 히스토그램인데, 가장 높은 값을 갖는 부분은 빨간색으로 표시된 x = 15, y = 25 정도의 좌표로 대략 10,000 이상의 값을 갖습니다. 이 의미는 파란색이면서 초록색인 픽셀의 개수가 가장 많다는 의미입니다. 중간과 오른쪽 그림을 봐도 초록과 파랑의 수치가 높은 것을 알 수 있습니다. 2차원 히스토그램의 의미는 x축이면서 y축인 픽셀의 분포를 알 수 있다는 것입니다. 논리 연산의 AND 연산과 같습니다.

<br>



## 6. 역투영

**2차원 히스토그램과 HSV 컬러 스페이스를 이용하면 색상으로 특정 물체나 사물의 일부분을 배경에서 분리할 수 있습니다.** 기본 원리는 물체가 있는 관심영역의 H와 V값의 분포를 얻어낸 후 전체 영상에서 해당 분포의 픽셀만 찾아내는 것입니다. 다음 예제에서는 마우스로 선택한 특정 물체만 배경에서 분리해 내는 모습을 보여주고 있습니다.

```py
'''마우스로 선택한 영역의 물체 배경 제거'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

win_name = 'back_projection'
img = cv2.imread('./img/pump_horse.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
draw = img.copy()

#--⑤ 역투영된 결과를 마스킹해서 결과를 출력하는 공통함수
def masking(bp, win_name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(bp,-1,disc,bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(win_name, result)

#--⑥ 직접 구현한 역투영 함수
def backProject_manual(hist_roi):
    #--⑦ 전체 영상에 대한 H,S 히스토그램 계산
    hist_img = cv2.calcHist([hsv_img], [0,1], None,[180,256], [0,180,0,256])
    #--⑧ 선택영역과 전체 영상에 대한 히스토그램 그램 비율계산
    hist_rate = hist_roi/ (hist_img + 1)
    #--⑨ 비율에 맞는 픽셀 값 매핑
    h,s,v = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]

    bp = np.minimum(bp, 1)
    bp = bp.reshape(hsv_img.shape[:2])
    cv2.normalize(bp,bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    #--⑩ 역 투영 결과로 마스킹해서 결과 출력
    masking(bp,'result_manual')
 
# OpenCV API로 구현한 함수 ---⑪ 
def backProject_cv(hist_roi):
    # 역투영 함수 호출 ---⑫
    bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi,  [0, 180, 0, 256], 1)
    # 역 투영 결과로 마스킹해서 결과 출력 ---⑬ 
    masking(bp,'result_cv')

# ROI 선택 ---①
(x,y,w,h) = cv2.selectROI(win_name, img, False)
if w > 0 and h > 0:
    #roi = draw[y:y+h, x:x+w]
    roi = img[y:y+h, x:x+w]
    cv2.rectangle(draw, (x, y), (x+w, y+h), (0,0,255), 2)
    #--② 선택한 ROI를 HSV 컬러 스페이스로 변경
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #--③ H,S 채널에 대한 히스토그램 계산
    hist_roi = cv2.calcHist([hsv_roi],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    #--④ ROI의 히스토그램을 매뉴얼 구현함수와 OpenCV 이용하는 함수에 각각 전달
    backProject_manual(hist_roi)
    backProject_cv(hist_roi)
cv2.imshow(win_name, draw)
cv2.waitKey()
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204106946-ede73d24-62bd-476e-9c1d-2b8be831fe6a.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204106950-f329796d-9879-4016-acd0-30ca182fa4e8.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204106953-7da2ac1e-65b8-4cdc-811f-9b62fb4f5c25.png">
</p>

코드 ①에서 마우스로 ROI를 선택하게 합니다. ROI를 선택하고 스페이스나 엔터 키를 누르면 코드 ②에서 선택한 관심영역을 HSV컬러 스페이스로 변경하고, 코드 ③에서 H와 S채널에 대한 2차원 히스토그램을 계산한 결과를 직접 구현한 함수와 OpenCV를 이용한 함수에 인자로 전달합니다.

먼저 코드 ⑥의 직접 구현한 함수를 살펴보면, 코드 ⑧에서 전달된 관심영역의 히스토그램을 전체 영상의 히스토그램으로 나누어 비율을 구합니다. 이때 1을 더한 이유는 분모가 0이 되어 오류가 발생하는 일이 없게 하기 위해서입니다. 비율을 구한다는 것은 관심영역과 비슷한 색상 분포를 갖는 히스토그램은 1에 가까운 값을 갖고 그 반대는 0 또는 0에 가까운 값을 갖게 되는 것으로 마스킹에 사용하기 좋다는 뜻입니다. 코드 ⑨는 이렇게 구한 비율을 원래 영상의 H와 S 픽셀 값에 매핑합니다. 여기서 `bp = hist_rate[h.ravel(), s.ravel()]` 가 핵심적인 코드입니다. hist_rate는 히스토그램 비율을 값으로 가지고 있고, 와 s는 실제 영상의 각 픽셀에 해당합니다. 따라서 H와 S가 교차되는 지점의 비율을 그 픽셀의 값으로 하는 1차원 배열을 얻게 됩니다. 여기서 사용한 NumPy 연산을 단순화시켜서 설명하면 아래의 코드와 같습니다.

```py
>>> v = np.arange(6).reshape(2,3)
>>> v
array([[0, 1, 2],
       [3, 4, 5]])
>>> row = np.array([1,1,1,0,0,0])
>>> col = np.array([0,1,2,0,1,2])
>>> v[row, col]
array([3, 4, 5, 0, 1, 2])
```

이렇게 얻는 값들은 비율이라서 1을 넘어서는 안 되므로 `np.minum(bp,1)` 로 1을 넘는 수는 1을 갖게 하고 나서 1차원 배열을 원래의 shape로 만들고 0~255 그레이 스케일에 맞는 픽셀 값으로 노멀라이즈합니다. 비율 연산 도중에 float 타입으로 변경된것을 unit8로 변경하면 작업은 끝나게 됩니다.

이런 복잡한 코드를 OpenCV는 아래와 같은 함수로 제공합니다.

* `cv2.calcBackProject(img, channel, hist, ranges, scale)`
    * `img` : 입력 영상, [img]처럼 리스트로 감싸서 표현
    * `channel` : 처리할 채널, 리스트로 감싸서 표현
        * 1채널: [0], 2채널: [0,1], 3채널: [0,1,2]
    * `hist` : 역투영에 사용할 히스토그램
    * `ranges` : 각 픽셀이 가질 수 있는 값의 범위
    * `scale` : 결과에 적용할 배율 계수

코드 ⑫에서 호출하는 `cv2.calcBackProject()` 함수는 세 번째 인자로 역투영에 사용할 히스토그램을 전달하면 역투영 결과를 반환합니다. 마지막 인자인 scale은 결과에 일정한 값을 계수로 적용할 수 있습니다.

코드 ⑤에 구현한 `masking()` 함수는 앞서 다룬 스레시홀드와 마스킹을 거쳐서 결과를 출력하는 함수인데, 여기에 함께 사용한 `cv2.getStructuringElement()` 와 `cv2.filter2D()` 함수는 마스크의 표면을 부드럽게 하기 위한 것입니다.

**역투영의 장점은 알파 채널이나 크로마 키 같은 보조 역할이 없어도 복잡한 모양의 사물을 분리할 수 있다는 것**입니다. **하지만 대상 사물의 색상과 비슷한 색상이 뒤섞여 있을 때는 효과가 지는 단점**도 있습니다.

<br>



## 7. 히스토그램 비교

**히스토그램은 영상의 픽셀 값의 분포를 갖는 정보이므로 이것을 비교하면 영상에 사용한 픽셀의 색상 비중이 얼마나 비슷한지 알 수 있습니다. 이것은 영상이 서로 얼마나 비슷한지를 알 수 있는 하나의 방법**입니다. OpenCV는 히스토그램을 비교해서그 유사도가 얼마인지 판단해 주는 함수를 아래와 같이 제공합니다.

* `cv2.compareHist(hist1, hist2, method)`
    * `hist1, hist2` : 비교할 2개의 히스토그램, 크기와 차원이 같아야 함
    * `method` : 비교 알고리즘 선택 플래그 상수
        * `cv2.HISTCMP_CORREL` : 상관관계 (1: 완전 일치, -1: 최대 불일치, 0: 무관계)
        * `cv2.HISTCMP_CHISQR` : 카이제곱 (0: 완전 일치, 큰 값(미정): 최대 불일치)
        * `cv2.HISTCMP_INTERSECT` : 교차(1: 완전 일치, 0: 최대 불일치(1로 정규화한경우))
        * `cv2.HISTCMP_BHATTACHARYYA` : 바타차야 (0: 완전 일치, 1: 최대 불일치)
        * `cv2.HISTCMP_HELLINGER` : HISTCMP_BHATTACHARYYA와 동일

이 함수는 첫 번째와 두 번째 인자에 비교하고자 하는 히스토그램을 전달하고, 마지막 인자에 어떤 플래그 상수를 전달하느냐에 따라 반환 값의 의미가 달라집니다. `cv2.HISTCMP_CORREL` 은 상관 관계를 기반으로 피어슨 상관계수로 유사성을 측정하고, `cv2.HISTCMP_CHISQR` 은 피어슨 상관계수 대신 카이제곱으로 유사성을 측정합니다. `cv2.HISTCMP_INTERSECT` 는 두 히스토그램의 교차점의 작은 값을 선택해서 그 합을 반환합니다. 반환 값을 원래의 히스토그램의 합으로 나누면 1과 0으로 노멀라이즈할 수 있습니다. `cv2.HISTCMP_BHATTACHARYYA` 는 두 분포의 중첩되는 부분을 측정합니다.

**서로 다른 영상의 히스토그램을 같은 조건으로 비교하기 위해서는 먼저 히스토그램을 노멀라이즈해야 합니다. 이미지가 크면 픽셀 수가 많고 당연히 히스토그램의 값도 더 커지기 때문**입니다.

다음 예제는 다른 각도에서 찍은 태권브이 장난감 이미지 3개와 코주부 박사 장난감을 찍은 이미지를 비교해서 각 비교 알고리즘에 다른 결과를 보여줍니다.

```py
'''히스토그램 비교'''
import cv2, numpy as np
import matplotlib.pylab as plt

img1 = cv2.imread('./img/taekwonv1.jpg')
img2 = cv2.imread('./img/taekwonv2.jpg')
img3 = cv2.imread('./img/taekwonv3.jpg')
img4 = cv2.imread('./img/dr_ochanomizu.jpg')

cv2.imshow('query', img1)
imgs = [img1, img2, img3, img4]
hists = []
for i, img in enumerate(imgs) :
    plt.subplot(1,len(imgs),i+1)
    plt.title('img%d'% (i+1))
    plt.axis('off') 
    plt.imshow(img[:,:,::-1])
    #---① 각 이미지를 HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #---② H,S 채널에 대한 히스토그램 계산
    hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
    #---③ 0~1로 정규화
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    hists.append(hist)


query = hists[0]
methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR, 
           'INTERSECT':cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}
for j, (name, flag) in enumerate(methods.items()):
    print('%-10s'%name, end='\t')
    for i, (hist, img) in enumerate(zip(hists, imgs)):
        #---④ 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
        ret = cv2.compareHist(query, hist, flag)
        if flag == cv2.HISTCMP_INTERSECT: # 교차 분석인 경우 
            ret = ret/np.sum(query)       # 비교대상으로 나누어 1로 정규화
        print("img%d:%7.2f"% (i+1 , ret), end='\t')
    print()

plt.show()

[output]
CORREL          img1:   1.00    img2:   0.70    img3:   0.56    img4:   0.23
CHISQR          img1:   0.00    img2:  67.34    img3:  35.71    img4:1129.50
INTERSECT       img1:   1.00    img2:   0.54    img3:   0.40    img4:   0.18
BHATTACHARYYA   img1:   0.00    img2:   0.48    img3:   0.47    img4:   0.79
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204107060-be5d2d41-b723-49f6-9529-6372478b10a4.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204107062-26e85000-df90-4486-a8a2-3bc057270fa5.png">
</p>

코드 ①, ②, ③은 각 영상을 HSV 컬러 스페이스로 바꾸고 H와 V에 대해 2차원 히스토그램을 계산해서 0~1로 노멀라이즈합니다. 코드 ④에서 각각의 비교 알고리즘을 이용해서 각 영상을 차례대로 비교합니다. 이때 `cv2.HISTCMP_INTERSECT` 인 경우 비교 원본의 히스토그램으로 나누기를 하면 0~1로 노멀라이즈할 수 있고 그러면 결과를 판별하기가 편리합니다.

img1과의 비교 결과는 모두 완전 일치를 보여주고 있으며, img4의 경우 가장 멀어진 값으로 나타나는 것을 확인할 수 있습니다.




