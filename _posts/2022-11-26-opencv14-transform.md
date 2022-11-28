---
layout: post
title: OpenCV geometric transform 이동, 확대/축소, 회전
category: OpenCV
tag: OpenCV
---

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 이동, 확대/축소, 회전

**기하학적 변환(geometric transform)**은 **영상의 좌표에 기하학적인 연산을 가해서 변환된 새로운 좌표를 얻는 것**입니다. **영상에 기하학적 변환을 하면 이동, 확대, 축소, 회전 등 일상생활에서 흔히 접하는 변환에서부터 볼록 거울에 비친 모습이나 일렁이는 물결에 비친 모습과 같은 여러 가지 왜곡된 모양으로도 변환할 수 있습니다.**

**영상의 기하학적 변환은 기존의 영상을 원하는 모양이나 방향 등으로 변환하기 위해 각 픽셀을 새로운 위치로 옮기는 것이 작업의 대부분**입니다. 그러기 위해서는 각 픽셀의 $x$, $y$ 좌표에 대해 옮기고자 하는 새로운 좌표 $x'$, $y'$ 을 구하는 연산이 필요합니다. 그러려면 픽셀 전체를 순회하면서 각 좌표에 대해 연산식을 적용해서 새로운 좌표를 구해야 하는데, 이때 사용할 **연산식을 가장 효과적으로 표현하는 방법이 행렬식**입니다.



## 1. 이동

**2차원 공간에서 물체를 다른 곳으로 이동시키려면 원래 있던 좌표에 이동시키려는 거리만큼 더해서 이동할 새로운 좌표를 구하면 됩니다.**

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204175394-fd911421-fa4f-4777-8d23-b9996862d9d5.png">
</p>

위 그림은 물고기 그림을 오른쪽 위로 이동하는 모습을 표현하고 있습니다. 이 그림에서 물고기의 어떤 점 $p(x, y)$를 $d_x$ 와 $d_y$ 만큼 옮기면 새로운 위치의 좌표 $p(x', y')$ 을 구할 수 있습니다. 이것을 수식으로 작성하면 아래와 같습니다.

$$
x' = x + d_x
y' = y + d_y
$$

위 방정식을 행렬식으로 바꾸어 표현하면 아래와 같습니다.

$$\begin{bmatrix} 
   x'  \\
   y'
\end{bmatrix}=
\begin{bmatrix}
1 & 0 & d_x  \\
0 & 1 & d_y
\end{bmatrix}
\begin{bmatrix}
x  \\
y  \\
1
\end{bmatrix}
$$

위의 행렬식을 아래와 같이 풀어서 표현하면 원래의 방정식과 같다는 것을 알 수 있습니다.

$$\begin{bmatrix} 
   x'  \\
   y'
\end{bmatrix}=
\begin{bmatrix}
x + d_x  \\
y + d_y
\end{bmatrix}
\begin{bmatrix}
1x + 0y + 1d_x  \\
0x + 1y + 1d_y
\end{bmatrix}
$$

여기서 굳이 행렬식을 언급하는 이유는, **좌표를 변환하는 과정은 OpenCV가 알아서 해주지만 어떻게 변환할 것인지는 개발자가 표현해야 하는데, 변환할 방정식을 함수에 전달할 때 행렬식이 표현하기 훨씬 더 적절하기 때문입니다.** 행렬식 중에서도 $x$, $y$ 는 이미 원본 이미지의 좌표 값으로 제공되므로 **$2 \times 3$ 변환행렬만 전달하면 연산이 가능합니다.** OpenCV는 $2 \times 3$ 행렬로 영상의 좌표를 변환시켜 주는 함수를 다음과 같이 제공합니다.

* `dst = cv2.warpAffine(src, mtrx, dsize [, dst, flags, borderMode, borderValue])`
    * `src` : 원본 영상, NumPy 배열
    * `mtrx` : 2 × 3 변환행렬, NumPy 배열, dtype = float32
    * `dsize` : 결과 이미지 크기, tuple(width, height)
    * `flags` : 보간법 알고리즘 선택 플래그
        * `cv2.INTER_LINEAR` : 기본 값, 인접한 4개 픽셀 값에 거리 가중치 사용
        * `cv2.INTER_NEAREST` : 가장 가까운 픽셀 값 사용
        * `cv2.INTER_AREA` : 픽셀 영역 관계를 이용한 재샘플링
        * `cv2.INTER_CUBIC` : 인접한 16개 픽셀 값에 거리 가중치 사용
        * `cv2.INTER_LANCZOS4` : 인접한 8개 픽셀을 이용한 란초의 알고리즘
    * `borderMode` : 외곽 영역 보정 플래그
        * `cv2.BORDER_CONSTANT` : 고정 색상 값(`999 | 12345 | 999`)
        * `cv2.BORDER_REPLICATE` : 가장 자리 복제 (`111 | 12345 | 555`)
        * `cv2.BORDER_WRAP` : 반복(`345 | 12345 | 123`)
        * `cv2.BORDER_REFLECT` : 반사(`321 | 12345 | 543`)
    * `borderValue` : `cv2.BORDER_CONSTANT` 의 경우 사용할 색상 값(기본값 = 0)
    * `dst` : 결과 이미지, NumPy 배열

`cv2.warpAffine()` 함수는 src 영상을 mtrx 행렬에 따라 변환해서 dsize 크기로 만들어서 반환합니다. 그뿐만 아니라 변환에 대부분 나타나는 픽셀 탈락 현상을 보정해주는 보간법 알고리즘과 경계 부분의 보정 방법도 선택할 수 있습니다. 다음 예제는 `cv2.warpAffine()` 함수와 변환행렬을 이용해서 영상을 이동 변환하는 예제입니다.

```py
'''평행 이동'''
import cv2
import numpy as np

img = cv2.imread('./img/fish.jpg')
rows, cols = img.shape[0:2] # 영상의 크기

dx, dy = 100, 50            # 이동할 픽셀 거리

# ---① 변환 행렬 생성 
mtrx = np.float32([[1, 0, dx],
                   [0, 1, dy]])  
# ---② 단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))   

# ---③ 탈락된 외곽 픽셀을 파랑색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
                        cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0) )

# ---④ 탈락된 외곽 픽셀을 원본을 반사 시켜서 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
                                cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow('original', img)
cv2.imshow('trans', dst)
cv2.imshow('BORDER_CONSTATNT', dst2)
cv2.imshow('BORDER_FEFLECT', dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204177804-f9af3e7b-6a13-4dbc-bc04-e655fc8742d7.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204177816-737a9df5-54b2-4b81-af74-023e32bbfbec.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204177825-54c31d41-7fda-4ff4-a193-4a48074c8b69.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204177838-ae5ca6b4-5f74-4e5f-992b-5d71e0d86fbd.png">
</p>

위 코드는 물고기 그림을 가로(x) 방향으로 100픽셀, 세로(y) 방향으로 50픽셀을 이동시키는 예제입니다. 코드 ①에서는 앞서 설명한 형식으로 변환행렬을 생성하고, 코드 ②에서는 `cv2.warpAffine()` 함수로 영상을 이동하게 만들었습니다. 이때 출력 영상의 크기를 원래의 크기보다 이동한 만큼 더 크게 지정해서 그림이 잘리지 않게 했는데, 영상의 좌측과 윗부분은 원래 없던 픽셀이 추가돼서 외곽 영역이 검게 표현됩니다. 코드 ⑧은 이 외곽 영역을 고정 값 파란색(255, 0, 0)으로 보정했으며, 코드 ④에서는 원본 영상을 거울에 비친 것처럼 복제해서 보정했습니다.

영상 이동에는 외곽 영역 이외에는 픽셀의 탈락이 발생하지 않으므로 이 예제에서 보간법 알고리즘을 선택하는 네 번째 인자는 의미가 없습니다.

<br>



## 2. 확대/축소

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204178050-4d92fdba-115d-4d5b-84ff-0ee2863e8307.png">
</p>

**영상을 확대 또는 축소하려면 원래 있던 좌표에 원하는 비율만큼 곱해서 새로운 좌표를 구할 수 있습니다.** 이때 확대/축소 비율을 가로와 세로 방향으로 각각 $\alpha$ 와 $\beta$ 라고 하면 변환행렬은 아래와 같습니다.

$$\begin{bmatrix} 
   x'  \\
   y'
\end{bmatrix}=
\begin{bmatrix}
\alpha & 0 & 0  \\
0 & \beta & 0
\end{bmatrix}
\begin{bmatrix}
x  \\
y  \\
1
\end{bmatrix}
$$

확대 혹은 축소를 하려면 $2 \times 2$ 행렬로도 충분히 표현이 가능한데 굳이 마지막 열에 0으로 채워진 열을 추가해서 $2 \times 3$ 행렬로 표현한 이유는 `cv2.warpAffine()` 함수와 이동 변환 때문입니다. 앞서 다룬 이동을 위한 행렬식은 $2 \times 3$ 행렬로 표현해야 하므로 여러 가지 기하학적 변환을 지원해야 하는 `cv2.warpAffine()` 함수는 $2 \times 3$ 행렬이 아니면 오류를 발생합니다. 행렬의 마지막 열에 $d_x$, $d_y$에 해당하는 값을 지정하면 확대와 축소뿐만 아니라 이동도 가능합니다.

```py
'''행렬을 이용한 확대와 축소'''
import cv2
import numpy as np

img = cv2.imread('./img/fish.jpg')
height, width = img.shape[:2]

# --① 0.5배 축소 변환 행렬
m_small = np.float32([[0.5, 0, 0],
                      [0, 0.5, 0]])  
# --② 2배 확대 변환 행렬
m_big = np.float32([[2, 0, 0],
                    [0, 2, 0]])  

# --③ 보간법 적용 없이 확대 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# --④ 보간법 적용한 확대 축소
dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), \
                        None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), \
                        None, cv2.INTER_CUBIC)

# 결과 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.imshow("small INTER_AREA", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204178689-68ec5344-f7c4-4227-b810-fb1952e727e2.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204178706-68a41885-77ca-4d92-aab9-d878f820174c.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204178719-323463c3-e73b-431d-8f9e-8fee5f574741.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204178735-f3e21f6a-5b0a-4e98-a182-a5e9378e91ff.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204178742-62362999-b61d-4dc0-bdb2-9b0ffcd9a660.png">
</p>

위 코드는 변환행렬을 이용해서 0.5배 축소와 2배 확대를 하는 예제입니다. 코드①과 ②에서 각각 축소와 확대에 필요한 변환행렬을 생성한 다음, 코드 ③에서는 보간법 알고리즘을 따로 지정하지 않았고, 코드 ④에서는 보간법 알고리즘을 따로 지정했습니다. **보간법 알고리즘으로는 축소에는 `cv2.INTER_AREA` 가 효과적이고, 확대에는 `cv2.INTER_CUBIC` 과 `cv2.INTER_LINEAR` 가 효과적인 것으로 알려져 있습니다.**

OpenCV는 변환행렬을 작성하지 않고도 확대와 축소 기능을 사용할 수 있게 `cv2.resize()` 함수를 별도로 제공합니다.

* `dst = cv2.resize(src, dsize, dst, fx, fy, interpolation)`
    * `src` : 입력 영상, NumPy 배열
    * `dsize` : 출력 영상 크기(확대/축소 목표 크기), 생략하면 fx, fy를 적용
        * (width, height)
    * `fx, fy` : 크기 배율, 생략하면 dsize를 적용
    * `interpolation` : 보간법 알고리즘 선택 플래그(`cv2.warpAffine()`과 동일)
    * `dst` : 결과 영상, NumPy 배열

`cv2.resize()` 함수는 확대 혹은 축소할 때 몇 픽셀로 할지 아니면 몇 퍼센트로 할지 선택할 수 있습니다. dsize로 변경하고 싶은 픽셀 크기를 직접 지정하거나 fx와 fy로 변경할 배율을 지정할 수 있습니다. 만약 dsize와 fx, fy 모두 값을 전달하면 dsize만 적용합니다.

```py
'''cv2.resize()로 확대와 축소'''
import cv2
import numpy as np

img = cv2.imread('./img/fish.jpg')
height, width = img.shape[:2]

#--① 크기 지정으로 축소
#dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)),\
#                        None, 0, 0, cv2.INTER_AREA)
dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), \
                         interpolation=cv2.INTER_AREA)

#--② 배율 지정으로 확대
dst2 = cv2.resize(img, None,  None, 2, 2, cv2.INTER_CUBIC)

#--③ 결과 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204179392-38394d70-83e2-48f4-beb6-006774c30d1e.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204179404-e2f5dc76-ba1d-4f46-957c-9224e6cdf2d2.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204179419-64fce8b6-6513-435f-ab8b-e30a51a5b0c8.png">
</p>

코드 ①에서는 원본 크기의 0.5배를 곱한 후 결과 크기를 구해서 전달하고 있으며, 배율은 None으로 처리했습니다. 반대로, 코드 ②에서는 크기 인자를 None으로 처리했고 배율을 각각 두 배로 전달합니다. **`cv2.resize()` 함수가 변환행렬을 이용하는 코드보다 사용하기 쉽고 간결한 것을 알 수 있습니다.**

<br>



## 3. 회전

**영상을 회전하려면 삼각함수를 써야 합니다.**

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204179839-4fb563f0-a32a-4fb0-97a5-8c385abdd816.png">
</p>

위 그림에서 **$p$ 라는 점을 원을 따라 $p'$ 으로 옮기는 것**을 **회전**이라고 하고, 그러기 위해서는 당연히 새로운 점 $p'$ 의 좌표 $x'$, $y'$ 을 구해야 합니다. 좌표를 구하기 전에 미리 정리해 둘 것이 있는데, 원의 반지름은 원 어디서나 동일하므로 원점 $O$ 와 $p$ 의 거리는 원점 $O$ 와 $p'$ 의 거리와 같고 그 값이 바로 원래 있던 점 $p$ 의 $x$ 좌표라는 것입니다.

이제 새로운 점 $p'$ 의 $x'$ 좌표를 구하기 위해 원 안에 가상의 직각삼각형을 그려보면 $\theta$ 각에 따라 변 $\overline{Op'}$ 와 변 $\overline{Ox'}$ 의 비율은 $cos\theta$ 임을 알 수 있습니다. 같은 방법으로 좌표 $y'$ 는 원 안의 직각삼각형의 변 $\overline{p'x'}$ 와 같으므로 변 $\overline{Op'}$ 와의 비율을 나타내는 $sin\theta$ 임을 알 수 있습니다. 변 $\overline{Op'}$ 는 원래의 좌표 $x$ 와 같으므로 새로운 점의 좌표는 $p'(x cos\theta, x sin\theta)$ 입니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204182688-96a86092-2ceb-45dc-8091-c37dc467cd4d.png">
</p>

회전은 원을 중심으로 진행되므로 위의 경우도 따져봐야 합니다. 이 경우도 원래의 점 $p$ 에서 원을 따라 회전한 $p'$ 의 좌표 $x'$, $y'$ 를 구해야 하는데, 이것도 이전과 같이 원 안의 직각삼각형으로 설명할 수 있습니다. 결국 새로운 점의 좌표는 $p'( -y sin\theta, y cos\theta)$ 입니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204182812-d6280783-47d2-4c21-937f-56ba35ba1c64.png">
</p>

위 그림은 위 두 경우의 수가 모두 반영된 모습을 보여주고 있으며, 이것을 행렬식으로 표현하면 다음과 같습니다.

$$\begin{bmatrix} 
   x'  \\
   y'
\end{bmatrix}=
\begin{bmatrix}
cos\theta & -sin\theta & 0  \\
sin\theta & cos\theta & 0
\end{bmatrix}
\begin{bmatrix}
x  \\
y  \\
1
\end{bmatrix}
$$

```py
'''변환행렬로 회전'''
import cv2
import numpy as np

img = cv2.imread('./img/fish.jpg')
rows, cols = img.shape[0:2]

# ---① 라디안 각도 계산(60진법을 호도법으로 변경)
d45 = 45.0 * np.pi / 180    # 45도
d90 = 90.0 * np.pi / 180    # 90도

# ---② 회전을 위한 변환 행렬 생성
m45 = np.float32( [[ np.cos(d45), -1* np.sin(d45), rows//2],
                    [np.sin(d45), np.cos(d45), -1*cols//4]])
m90 = np.float32( [[ np.cos(d90), -1* np.sin(d90), rows],
                    [np.sin(d90), np.cos(d90), 0]])

# ---③ 회전 변환 행렬 적용
r45 = cv2.warpAffine(img, m45, (cols,rows))
r90 = cv2.warpAffine(img, m90, (rows,cols))

# ---④ 결과 출력
cv2.imshow("origin", img)
cv2.imshow("45", r45)
cv2.imshow("90", r90)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204183020-8acdd72e-99e8-40c4-ba13-2470d06d95e5.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204183037-36bcd6a0-4620-46f8-b6ba-b57cdccde091.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204183049-895caa8e-ffb2-4116-9a68-1abc24ac6656.png">
</p>

코드 ①은 변환행렬에 사용할 회전 각을 60진법에서 라디안(radian)으로 변경합니다. 코드 ②에서 변환행렬을 생성하는데, 삼각함수는 NumPy의 `np.cos()`, `np.sin()` 함수를 사용했습니다. 변환행렬의 마지막 열에 0이 아닌 `rows//2`, `-1*cols//4`, `rows` 를 사용한 이유는 영상의 회전 기준 축이 좌측 상단이 되므로 회전한 영상은 보여지는 영역 바깥으로 벗어나게 돼서 좌표를 가운데로 옮기기 위한 것으로 회전 축을 지정하는 효과와 같습니다. 변환행렬의 마지막 열을 이동에 사용한다는 내용은 앞서 다루었습니다.

회전을 위한 변환행렬 생성은 다소 까다로운 데다가 회전 축까지 반영하려면 일이 조금 복잡해집니다. OpenCV는 개발자가 복잡한 계산을 하지 않고도 변환행렬을 생성할 수 있게 아래와 같은 함수를 제공합니다.

* `mtrx = cv2.getRotationMatrix2D(center, angle, scale)`
    * `center` : 회전 축 중심 좌표, 튜플(x, y)
    * `angle` : 회전 각도, 60진법
    * `scale` : 확대/축소 배율

이 함수를 쓰면 중심축 지정과 확대/축소까지 반영해서 손쉽게 변환행렬을 얻을 수있습니다.

```py
'''회전 변환행렬 구하기'''
import cv2

img = cv2.imread('./img/fish.jpg')
rows,cols = img.shape[0:2]

#---① 회전을 위한 변환 행렬 구하기
# 회전축:중앙, 각도:45, 배율:0.5
m45 = cv2.getRotationMatrix2D((cols/2,rows/2), 45, 0.5) 
# 회전축:중앙, 각도:90, 배율:1.5
m90 = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1.5) 

#---② 변환 행렬 적용
img45 = cv2.warpAffine(img, m45,(cols, rows))
img90 = cv2.warpAffine(img, m90,(cols, rows))

#---③ 결과 출력
cv2.imshow('origin',img)
cv2.imshow("45", img45)
cv2.imshow("90", img90)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204183882-9e6d449a-f233-4692-a0dc-91b629eca516.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204183889-ed69e44d-a074-4eb8-a89f-602aa3b5154d.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204183894-ef797771-5f99-47e5-b414-d5d681ae2e18.png">
</p>




