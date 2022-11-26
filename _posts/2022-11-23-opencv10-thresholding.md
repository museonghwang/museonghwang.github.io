---
layout: post
title: OpenCV Image Processing 컬러 스페이스
category: OpenCV
tag: OpenCV
---

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 스레시홀딩

이미지를 검은색과 흰색만으로 표현한 것을 바이너리(binary, 이진화) 이미지라고 합니다. 이렇게 하는 이유는 **이미지에서 원하는 피사체의 모양을 좀 더 정확히 판단하기 위해**서입니다. 예를 들면, 종이에서 글씨만을 분리하거나 배경에서 전경을 분리하는 것과 같은 작업입니다.

**스레시홀딩(thresholding)**이란 여러 점수를 커트라인을 기준으로 합격과 불합격으로 나누는 것처럼 **여러 값을 경계점을 기준으로 두 가지 부류로 나누는 것**으로, 바이너리 이미지를 만드는 가장 대표적인 방법입니다.



## 1. 전역 스레시홀딩

바이너리 이미지를 만들기 위해서는 컬러 이미지를 그레이 스케일로 바꾸고 각 픽셀의 값이 경계 값을 넘으면 255, 넘지 못하면 0을 지정합니다. 이런 작업은 간단한 NumPy 연산만으로도 충분히 할 수 있지만, OpenCV는 `cv2.threshold()` 함수로 더 많은 기능을 제공합니다.

다음 코드는 NumPy 연산과 OpenCV 함수로 각각 바이너리 이미지를 만드는 과정을 보여줍니다.

```py
'''바이너리 이미지 만들기'''
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('./img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE) #이미지를 그레이 스케일로 읽기

# --- ① NumPy API로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img)   # 원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[img > 127] = 255      # 127 보다 큰 값만 255로 변경

# ---② OpenCV API로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
print(ret)  # 127.0, 바이너리 이미지에 사용된 문턱 값 반환

# ---③ 원본과 결과물을 matplotlib으로 출력
imgs = {'Original': img, 'NumPy API': thresh_np, 'cv2.threshold': thresh_cv}
for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()

[output]
127.0
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204090640-fe1b1b5d-da46-4836-8a01-bafe36a7ff99.png">
</p>

위 예제는 검은색에서 흰색으로 점점 변하는 그러데이션 이미지를 그레이 스케일로 읽어서 바이너리 이미지를 만드는 예제입니다. 코드 ①에서 원본 이미지와 같은 크기이면서 O(zero)으로 채워진 NumPy 배열을 생성하고 나서 127보다 큰 값을 갖는 요소에 255를 할당하는 연산을 해서 바이너리 이미지로 만들고 있습니다.

코드 ②는 `cv2.threshold()` 함수를 이용해서 간단히 바이너리 이미지를 만들고 있습니다. 코드 ③은 각각 생성한 바이너리 이미지와 원본 이미지를 출력합니다. 이때 사용한 `cv2.threshold()` 함수의 사용법은 다음과 같습니다.

* `ret, out = cv2.threshold(img, threshold, value, type_flag)`
    * `img` : NumPy 배열, 변환할 이미지
    * `threshold` : 경계 값
    * `value` : 경계 값 기준에 만족하는 픽셀에 적용할 값
    * `type_flag` : 스레시홀드 적용 방법 지정
        * `cv2.THRESH_BINARY` : px > threshold ? value : 0, 픽셀 값이 경계 값을 넘으면 value를 지정하고, 넘지 못하면 0을 지정
        * `cv2.THRESH_BINARY_INV` : px > threshold ? 0 : value, cv2.THRESH_BINARY의 반대
        * `cv2.THRESH_TRUNC` : px > threshold ? value : px, 픽셀 값이 경계 값을 넘으면 value를 지정하고, 넘지 못하면 원래의 값 유지
        * `cv2.THRESH_TOZERO` : px > threshold ? px : 0, 픽셀 값이 경계 값을 넘으면 원래 값을 유지, 넘지 못하면 0을 지정
        * `cv2.THRESH_TOZERO_INV` : px > threshold ? 0 : px, cv2.THRESH_TOZERO의 반대
    * `ret` : 스레시홀딩에 사용한 경계 값
    * `out` : 결과 바이너리 이미지

이 함수의 반환 값은 튜플로 2개의 값을 반환하는데, 첫 번째 항목은 스레시홀딩에 사용한 경계 값이고, 두 번째 항목은 스레시홀딩된 바이너리 이미지입니다. 대부분의 경우 첫 번째 반환 항목인 ret는 threshold 인자로 전달한 값과 같아서 쓸모 없습니다.

이 함수는 단순히 경계 값에 따라 0과 255로 나누는 `cv2.THRESH_BINARY` 말고도 몇 가지 기능의 플래그 상수를 사용할 수 있게 해줍니다. 다음 예제에서는 위에 나열한 몇 가지 다른 플래그를 이용한 스레시홀딩을 사례로 보여줍니다. 코드와 실행 결과만 봐도 쉽게 이해할 수 있을 것입니다.

```py
'''스레시홀딩 플래그 실습'''
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('./img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)

_, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, t_bininv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, t_truc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, t_2zr = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, t_2zrinv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imgs = {'origin':img, 'BINARY':t_bin, 'BINARY_INV':t_bininv, \
        'TRUNC':t_truc, 'TOZERO':t_2zr, 'TOZERO_INV':t_2zrinv}
        
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])
    
plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204091162-7a43c682-028f-4c38-b0d5-335e6402ef07.png">
</p>

<br>



## 2. 오츠의 알고리즘

**바이너리 이미지를 만들 때 가장 중요한 작업은 경계 값을 얼마로 정하느냐입니다.** 종이에 출력한 문서를 바이너리 이미지로 만드는 것을 예를 들면, 새하얀 종이에 검은색으로 출력된 문서의 영상이라면 굳이 스레시홀드를 적용할 필요가 없습니다. 하지만, 현실은 흰색, 누런색, 회색 종이에 검은색, 파란색 등으로 인쇄된 문서가 더 많기 때문에 **적절한 경계 값을 정하기 위해서는 여러 차례에 걸쳐 경계 값을 조금씩 수정해 가면서 가장 좋은 경계 값을 찾아야 합니다.**

```py
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('./img/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE) #이미지를 그레이 스케일로 읽기
thresholds = [80, 100, 120, 140, 150, 170, 190]
imgs = {'Original' : img}
for t in thresholds:
    _, t_img = cv2.threshold(img, t, 255, cv2.THRESH_BINARY) 
    imgs['t:%d'%t] = t_img

for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(2, 4, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204091711-dc2309c1-00e1-4868-98e9-33c1a0d552ba.png">
</p>

위 그림은 오래된 신문 기사를 스크랩해서 스캔한 영상인데, 왼쪽 처음 그림이 원본이고 경계 값을 80부터 20씩 증가시키면서 스레시홀딩한 결과입니다. 결과를 살펴보면 아마도 글씨와 그림을 가장 잘 얻을 수 있는 경계 값은 120과 140 사이쯤인 것을 알 수 있습니다. 그러니까 다음번에 시도해 볼 경계 값은 130 정도가 적당해 보입니다. 이와 같이 반복적인 경계 값 찾기 시도는 귀찮고 시간도 많이 걸립니다.

1979년 오츠 노부유키(Nobuyuki Otsu)는 반복적인 시도 없이 한 번에 효율적으로 경계 값을 찾을 수 있는 방법을 제안했는데, 그의 이름을 따서 그것을 **오츠의 이진화 알고리즘(Otsu's binarization method)**이라고 합니다. **오츠의 알고리즘은 경계값을 임의로 정해서 픽셀들을 두 부류로 나누고 두 부류의 명암 분포를 반복해서 구한 다음 두 부류의 명암 분포를 가장 균일하게 하는 경계 값을 선택합니다.** 이것을 수식으로 표현하면 다음과 같습니다.

$$
\sigma^2_w(t) = w_1(t)\sigma^2_1(t) + w_2(t)\sigma^2_2(t)
$$
* $t$ : 0~255, 경계 값
* $w_1, w_2$ : 각 부류의 비율 가중치
* $\sigma^2_1(t), \sigma^2_2(t)$ : 각 부류의 분산

<br>

OpenCV는 이미 구현한 오츠의 알고리즘을 사용할 수 있게 제공해 주는데, 이것을 사용하려면 앞서 설명한 `cv2.threshold()` 함수의 마지막 인자에 `cv2.THRESH_OTSU` 를 추가해서 전달하기만 하면 됩니다. 그러면 원래 경계 값을 전달해야 하는 두 번째 인자 threshold는 무시되므로 아무 숫자나 전달해도 되고, 실행 후 결과 값으로 오츠의 알고리즘에 의해 선택된 경계 값은 반환 값 첫 번째 항목 ret로 받을 수 있습니다. 아래 코드는 `cv2.threshold()` 함수에 오츠의 알고리즘을 적용하는 코드입니다.

```py
ret, t_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

마지막 플래그에는 앞서 설명한 스레시홀드 방식을 결정하는 플래그와 파이프('|') 문자로 연결하여 전달합니다.

```py
'''오츠의 알고리즘을 적용한 스레시홀드'''
import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지를 그레이 스케일로 읽기
img = cv2.imread('./img/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE) 

# 경계 값을 130으로 지정  ---①
_, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)        

# 경계 값을 지정하지 않고 OTSU 알고리즘 선택 ---②
t, t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
print('otsu threshold:', t)     # Otsu 알고리즘으로 선택된 경계 값 출력

imgs = {'Original': img, 't:130':t_130, 'otsu:%d'%t: t_otsu}
for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()

[output]
otsu threshold: 131.0
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204091996-e771623e-edd5-45db-82d4-0d3f8f473662.png">
</p>

위 코드 ①에서는 기존에 여러 번 시도해서 알아낸 경계 값 130을 직접 지정해서 바이너리 이미지를 얻습니다. 반면에, 코드 ②에서는 오츠의 알고리즘을 적용하고 경계 값으로는 의미 없는 -1을 전달했더니 결과적으로 경계 값 131을 자동으로 계산해서 반환하고 적절한 바이너리 이미지를 얻게 됐습니다. 사람이 여러 번 시도해서 얻는 값과 거의 비슷한 것을 알 수 있습니다. **하지만, 오츠의 알고리즘은 모든 경우의 수에 대해 경계 값을 조사해야 하므로 속도가 빠르지 못하다는 단점**이 있습니다. **또한 노이즈가 많은 영상에는 오츠의 알고리즘을 적용해도 좋은 결과를 얻지 못하는 경우가 많은데, 이때는 블러링 필터를 먼저 적용해야 합니다.**

<br>



## 3. 적응형 스레시홀드

원본 영상에 조명이 일정하지 않거나 배경색이 여러 가지인 경우에는 아무리 여러번 경계 값을 바꿔가며 시도해도 하나의 경계 값을 이미지 전체에 적용해서는 좋은 결과를 얻지 못합니다. 이때는 **이미지를 여러 영역으로 나눈 다음 그 주변 픽셀 값만 가지고 계산을 해서 경계 값을 구해야 하는데**, 이것을 **적응형 스레시홀드(adaptive threshold)**라고 합니다.

OpenCV에서는 이 기능을 다음 함수로 제공합니다.

* `cv2.adaptiveThreshold(img, value, method, type_flag, block_size, C)`
    * `img` : 입력 영상
    * `value` : 경계 값을 만족하는 픽셀에 적용할 값
    * `method` : 경계 값 결정 방법
        * `cv2.ADPTIVE_THRESH_MEAN_C` : 이웃 픽셀의 평균으로 결정
        * `cv2.ADPTIVE_THRESH_GAUSSIAN_C` : 가우시안 분포에 따른 가중치의 합으로 결정
    * `type_flag` : 스레시홀드 적용 방법 지정(`cv2.threshold()` 함수와 동일)
    * `block_size` : 영역으로 나눌 이웃의 크기($n \times n$), 홀수(3, 5, 7, ...)
    * `C` : 계산된 경계값 결과에서 가감할 상수(음수 가능)

```py
'''적응형 스레시홀드 적용'''
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

blk_size = 9        # 블럭 사이즈
C = 5               # 차감 상수 
img = cv2.imread('./img/sudoku.png', cv2.IMREAD_GRAYSCALE) # 그레이 스케일로  읽기

# ---① 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# ---② 어뎁티드 쓰레시홀드를 평균과 가우시안 분포로 각각 적용
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, blk_size, C)

# ---③ 결과를 Matplot으로 출력
imgs = {'Original': img, 'Global-Otsu:%d'%ret:th1, \
        'Adapted-Mean':th2, 'Adapted-Gaussian': th3}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2,i+1)
    plt.title(k)
    plt.imshow(v,'gray')
    plt.xticks([]),plt.yticks([])

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204092326-7dd5f748-1b95-45a4-a406-832d20e460f4.png">
</p>

위 코드 ①은 오츠의 알고리즘을 적용해서 얻은 96을 경계 값으로 전체 이미지에 적용했지만 결과를 보면 좌측 하단은 검게 타버리고, 우측 상단은 하얗게 날아간 것을 알 수 있습니다. 반면에, 코드 ②에서는 적응형 스레시홀드를 평균 값과 가우시안 분포를 각각 적용해서 훨씬 좋은 결과를 얻을 수 있습니다. 그중에서도 가우시안 분포를 이용한 결과는 선명함은 떨어지지만 잡티(noise)가 훨씬 적은 것을 알 수 있습니다.

**경계 값을 전체 이미지에 적용하는 것을 전역적(global) 적용**이라고 하는 반면에, **이미지를 여러 구역으로 나누어 그 구역에 맞는 경계 값을 찾는 것을 지역적(local)적용**이라고 합니다. **대부분의 이미지는 조명 차이와 그림자 때문에 지역적 적용이 필요합니다.**




