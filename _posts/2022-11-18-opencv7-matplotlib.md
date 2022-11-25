---
layout: post
title: OpenCV를 위한 Matplotlib
category: OpenCV
tag: OpenCV
---

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# Matplotlib

Matplotlib은 파이썬에서 가장 인기 있는 데이터 시각화 라이브러리입니다. 이것을 이용하면 도표나 차트 등을 손쉽게 그릴 수 있습니다. 이미지 프로세싱이나 컴퓨터비전 분야에서는 여러 이미지를 화면에 띄우고 싶을 때 OpenCV의 `cv2.imshow()` 함수를 여러 번 호출하면 창이 여러 개 열리므로 한 화면에 여러 이미지를 띄우려는 단순한 이유로 Matplotlib를 사용하는 경우가 가장 많습니다. 물론 그뿐만 아니라 이미지로부터 각종 히스토그램 등의 통계 자료를 뽑아내어 그래프나 다이어그램으로 표시하는 용도로도 많이 사용합니다.

Matplotlib의 Pyplot 모듈은 다양한 종류의 도표를 빠르고 쉽게 생성할 수 있는 함수들을 모아놓은 모듈입니다.



## 1. plot

그래프를 그리는 가장 간단한 방법은 `plot()` 함수를 사용하는 것입니다.

1차원 배열을 인자로 전달하면 배열의 인덱스를 x 좌표로, 배열의 값을 y 좌표로써서 그래프를 그립니다. 아래의 코드는 가장 간단한 방법으로 그래프를 그리는 예제입니다.

```py
'''plot 그리기'''
import matplotlib.pyplot as plt
import numpy as np

a = np.array([2,6,7,3,12,8,4,5])    # 배열 생성
plt.plot(a)                         # plot 생성
plt.show()                          # plot 그리기
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202974533-aa46e121-b5f4-44d8-8d2d-54a34808b08b.png">
</p>

두 배열의 상관관계를 그래프로 표시하려면 `plot()` 함수의 인자로 배열을 순차적으로 전달하면 차례대로 x, y 좌표로 사용해서 그래프를 그립니다. 2개의 배열로 그래프로 표시하는 예시는 아래와 같습니다.

```py
'''y=x^2 그래프 그리기'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10) # 0,1,2,3,4,5,6,7,8,9
y = x**2          # 0,1,4,9,16,25,36,49,64,81
plt.plot(x,y)     # plot 생성       ---①
plt.show()        # plot 화면에 표시  ---②
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202974829-5f494f0b-5c50-4b35-8bfa-aea00c91ecb1.png">
</p>

위의 코드 ①에서 배열 x, y의 값으로 플롯을 생성하고, 코드 ②에서 화면에 표시합니다.



## 2. color와 style

그래프의 선에 색상과 스타일을 지정할 수 있습니다. `plot()` 함수의 마지막 인자에 아래의 색상 기호 중 하나를 선택해서 문자로 전달하면 색상이 적용됩니다.

* 색상 기호
    * b : 파란색(Blue)
    * g : 초록색(Green)
    * r : 빨간색(Red)
    * c : 청록색(Cyan)
    * m : 자홍색(Magenta)
    * y : 노란색(Yellow)
    * k : 검은색(black)
    * w : 흰색(White)

다음 코드는 선을 빨간색으로 표시한 예제입니다.

```py
'''plot의 색 지정'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10) # 0,1,2,3,4,5,6,7,8,9
y = x **2         # 0,1,4,9,16,25,36,49,64,81
plt.plot(x, y, 'r') # plot 생성 ---①
plt.show()        # plot 화면에 표시
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202975269-a30b3008-3740-4b4a-bd4f-0ff7d9eccdf2.png">
</p>

위의 코드 ①에서 `plot(x, y, 'r')` 로 빨간색을 지정하여 선을 그립니다.

색상과 함께 스타일도 지정할 수 있는데, 스타일 기호 중 하나를 색상 값에 이어 붙여서 사용합니다.

```py
'''다양한 스타일 지정'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
f1 = x * 5
f2 = x **2
f3 = x **2 + x*2

plt.plot(x,'r--')   # 빨강색 이음선
plt.plot(f1, 'g.')  # 초록색 점
plt.plot(f2, 'bv')  # 파랑색 역 삼각형
plt.plot(f3, 'ks' ) # 검정색 사각형
plt.show()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202975581-1ebed9f5-5a3e-4d39-940c-e0018959ada1.png">
</p>

앞의 코드에서 다양한 색상과 스타일의 그래프를 표시하고 있습니다.



## 3. subplot

앞서 살펴본 예제는 여러 배열 값을 이용해서 `plt.plot()` 함수를 여러 번 호출하면 하나의 다이어그램에 겹쳐서 그래프를 그렸습니다. 각각의 그래프를 분리해서 따로 그려야 할 때는 `plt.subplot()` 함수를 이용합니다. 이 함수는 3개의 인자를 이용해서 몇 행 몇 열로 분할된 그래프에 몇 번째 그래프를 그릴지를 먼저 지정한 후에 `plt.plot()` 함수를 호출하면 그 자리에 그래프를 그리게 됩니다.

```py
'''subplot'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)

plt.subplot(2,2,1)  # 2행 2열 중에 1번째
plt.plot(x,x**2)

plt.subplot(2,2,2)  # 2행 2열 중에 2번째
plt.plot(x,x*5)

plt.subplot(223)    # 2행 2열 중에 3번째
plt.plot(x, np.sin(x))

plt.subplot(224)    # 2행 2열 중에 4번째
plt.plot(x,np.cos(x))

plt.show()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202975827-39a0e646-3033-4182-a2c6-bb4952ae657a.png">
</p>

`subplot(2,2,1)` 처럼 3개의 인자를 전달하는 것과 `subplot(221)` 처럼 세 자리 숫자 한개를 전달하는 것은 똑같이 동작합니다.



## 4. 이미지 표시

`plt.plot()` 대신에 `plt.imshow()` 함수를 호출하면 OpenCV로 읽어들인 이미지를 그래프 영역에 출력할 수 있습니다.

```py
'''plot으로 이미지 출력'''
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/wonyoung.jpg')

plt.imshow(img) # 이미지 표시
plt.show()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202976047-8edad987-0545-4076-9ca9-0a5e8835c6c8.png">
</p>

앞 예제의 실행 결과는 색상이 이상합니다. **`plt.imshow()` 함수는 컬러 이미지인 경우 컬러 채널을 R, G, B 순으로 해석하지만 OpenCV 이미지는 B, G, R순으로 만들어져서 색상의 위치가 반대라서 그렇습니다. 그래서 OpenCV로 읽은 이미지를 R, G, B 순으로 순서를 바꾸어서 `plt.imshow()` 함수에 전달해야 제대로 된 색상으로 출력할 수 있습니다.**

```py
'''컬러 채널을 변경한 이미지 출력'''
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/wonyoung.jpg')

plt.imshow(img[:,:,::-1])   # 이미지 컬러 채널 변경해서 표시 ---①
plt.xticks([])              # x좌표 눈금 제거 ---②     
plt.yticks([])              # y좌표 눈금 제거 ---③
plt.show()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202976365-80741134-8ef4-4b15-872f-0e2ebc2e6012.png">
</p>

위의 코드 ① `img[:,:,::-1]` 은 컬러 채널의 순서를 바꾸는 것인데, 이 코드의 의미는 이렇습니다. 3차원 배열의 모든 내용을 선택하는 것은 `img[:,:,:]` 입니다. 이때 마지막 축의 길이가 3이므로 다시 `img[:,:,::]` 로 바꾸어 쓸수 있습니다. 이때 마지막 축의 요소의 순서를 거꾸로 뒤집기 위해 `img[:,:,::−1]` 로 쓸 수 있습니다. 이것을 풀어서 작성하면 다음 두 가지 코드와 같습니다.

```py
img [:,:,(2,1,0)]
```

또는

```py
img[:,:,2], img[:,:,1], img[:,:,0] = img[:,:,0], img[:,:,0], img[:,:,2]
```

위의 코드 ②와 ③은 이미지 출력 결과 화면에 나타나는 x, y 좌표 눈금을 제거하기 위한 코드입니다. 단순히 이미지만 보여주려고 하는데 눈금이 신경 쓰이면 필요에 따라 적용하면 됩니다.

앞서 설명한 대로 프로그램의 결과로 이미지를 여러 개 출력해야 하는 경우, OpenCV의 `cv2.imshow()` 함수는 여러 번 호출하면 매번 새로운 창이 열리기 때문에 귀찮습니다. `plt.imshow()` 함수는 `plt.subplot()` 함수와 함께 사용하면 하나의 창에 여러개의 이미지를 동시에 출력할 수 있으니 이런 경우 좋은 대안이 될 수 있습니다.

```py
'''여러 이미지 동시 출력'''
import matplotlib.pyplot as plt
import numpy as np
import cv2

img1 = cv2.imread('./img/wonyoung.jpg')
img2 = cv2.imread('./img/wonyoung2.png')
img3 = cv2.imread('./img/wonyoung3.jpg')


plt.subplot(1,3,1)  # 1행 3열 중에 1번째
plt.imshow(img1[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1,3,2)  # 1행 3열 중에 2번째
plt.imshow(img2[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1,3,3)  # 1행 3열 중에 3번째
plt.imshow(img3[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.show()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202979242-963ac95a-aa64-4146-a1ea-4635f198bbb2.png">
</p>




