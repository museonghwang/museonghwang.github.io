---
layout: post
title: OpenCV Image Processing 이미지 연산
category: OpenCV
tag: OpenCV
---

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 이미지 연산

영상에 연산하는 방법을 알아봅니다. **연산 결과는 새로운 영상을 만들어 내므로 그 자체가 목적이 될 수도 있지만, 정보를 얻기 위한 과정일 수도 있습니다.**



## 1. 영상과 영상의 연산

영상에 연산을 할 수 있는 방법은 NumPy의 브로드캐스팅 연산을 직접 적용하는 방법과 OpenCV에서 제공하는 네 가지 함수를 사용하는 방법이 있습니다. OpenCV에서 굳이 연산에 사용할 함수를 제공하는 이유는 영상에서의 한 픽셀이 가질 수 있는 값의 범위는 0~255인데, 더하기나 빼기 연산을 한 결과가 255보다 클 수도 있고 0보다 작을 수도 있어서 결과 값을 0과 255로 제한할 안전 장치가 필요하기 때문입니다. OpenCV에는 아래의 함수로 이와 같은 기능을 제공합니다.

* `dest = cv2.add(src1, src2[, dest, mask, dtype])` : src1과 src2 더하기
    * `src1` : 입력 영상 1 또는 수
    * `src2` : 입력 영상 2 또는 수
    * `dest` : 출력 영상
    * `mask` : 0이 아닌 픽셀만 연산
    * `dtype` : 출력 dtype
* `dest = cv2.substract(src1, src2[, dest, mask, dtype])` : src1에서 src2를 빼기
    * 모든 인자는 `cv2.add()` 함수와 동일
* `dest = cv2.multiply(src1, src2[, dest, scale, dtype])` : src1과 src2를 곱하기
    * `scale` : 연산 결과에 추가 연산할 값
* `dest = cv2.divide(src1, src2[, dest, scale, dtype])` : src1을 src2로 나누기
    * 모든 인자는 `cv2.multiply()`와 동일

영상에 사칙 연산을 적용해서 그 차이를 알아봅니다.

```py
'''영상의 사칙 연산'''
import cv2
import numpy as np

# ---① 연산에 사용할 배열 생성
a = np.uint8([[200, 50]]) 
b = np.uint8([[100, 100]])

#---② NumPy 배열 직접 연산
add1 = a + b
sub1 = a - b
mult1 = a * 2
div1 = a / 3

# ---③ OpenCV API를 이용한 연산
add2 = cv2.add(a, b)
sub2 = cv2.subtract(a, b)
mult2 = cv2.multiply(a , 2)
div2 = cv2.divide(a, 3)

#---④ 각 연산 결과 출력
print(add1, add2)
print(sub1, sub2)
print(mult1, mult2)
print(div1, div2)

[output]
[[ 44 150]] [[255 150]]
[[100 206]] [[100   0]]
[[144 100]] [[255 100]]
[[66.66666667 16.66666667]] [[67 17]]
```

위 코드 ①에서 연산을 테스트할 대상으로 NumPy 배열을 생성합니다. 코드 ②는 사칙 연산자를 직접 사용했고, 코드 ③은 OpenCV의 4개의 함수를 이용했습니다. 코드 ④에서 결과를 각각 출력하고 있습니다.

출력 결과를 살펴보면 200과 100을 더하고 50과 100을 더한 결과가 각각 44 와 255, 150과 150으로 50과 100의 결과는 동일하게 나타납니다. 하지만, 200과 100을 더한 결과는 300인데, 더하기(+) 연산자로 직접 더한 결과는 255를 초과하는 44이고, `cv2.add()` 함수의 결과는 최대 값인 255입니다. 50에서 100을 빼는 연산은 -50인데, 마찬가지로 직접 빼기(-) 연산한 결과는 206으로 정상적이지 않지만, `cv2.subtract()` 함수의 결과는 최소 값인 0입니다. 곱하기와 나누기 연산도 OpenCV 함수의 결과는 255를 초과하지 않고 소수점 이하를 갖지 않습니다.

OpenCV의 네 가지 연산 함수 중에 `cv2.add()` 함수를 대표로 해서 좀 더 자세히 설명해 보겠습니다. 함수의 첫 번째와 두 번째 인자에는 연산의 대상을 NumPy 배열로 전달합니다. 그 두 인자를 더한 결과는 세 번째 인자로 전달한 배열에 할당하고 결과값으로 다시 반환합니다. 만약 `c = a + b` 와 같은 연산이 필요하다면 다음의 세 코드의 결과는 똑같습니다.

```py
c = cv2.add(a, b) 또는 c = cv2.add(a, b, None) 또는 cv2.add(a, b, c)
```

만약 `b += a` 와 같이 두 입력의 합산 결과를 입력 인자의 하나에 재할당하고 싶을 때는 다음의 두 코드와 같이 작성할 수 있고 결과는 같습니다.

```py
cv2.add(a, b, b) 또는 b = cv2.add(a, b)
```

하지만, 네 번째 인자인 mask를 지정하는 경우에는 얘기가 다릅니다. 네 번째 인자에 전달한 NumPy 배열에 어떤 요소 값이 0이면 그 위치의 픽셀은 연산을 하지 않습니다. 이때 세 번째 인자인 결과를 할당할 인자의 지정 여부에 따라 결과는 달라집니다. 코드로 예를 들어 보겠습니다.

```py
'''mask와 누적 할당 연산'''
import cv2
import numpy as np

#---① 연산에 사용할 배열 생성
a = np.array([[1, 2]], dtype=np.uint8)
b = np.array([[10, 20]], dtype=np.uint8)

#---② 2번째 요소가 0인 마스크 배열 생성 
mask = np.array([[1, 0]], dtype=np.uint8)

#---③ 누적 할당과의 비교 연산
c1 = cv2.add(a, b, None, mask)
print(c1)
c2 = cv2.add(a, b, b, mask)
print(c2, b)

[output]
[[11  0]]
[[11 20]] [[11 20]]
```

예제에서 a와 b의 더하기 연산은 1+10, 2+20 연산이 각각 이뤄져야 하지만, 네번째 인자인 mask의 두 번째 요소의 값이 0이므로 2+20의 연산은 이루어지지 않습니다. 따라서 c1의 결과는 11과 0입니다. 하지만, 누적 할당을 적용한 c2의 두 번째 항목은 b의 두 번째 항목인 20을 그대로 갖게 됩니다. 이때 주의할 것은 b도 c2와 동일하게 연산의 결과를 갖게 되는 것입니다. 만약 b의 값이 연산 전 상태를 그대로 유지되길 원한다면 아래와 같이 수정해서 사용할 수 있습니다.

```py
c2 = cv2.add(a, b, b.copy(), mask)
```

<br>



## 2. 알파 블렌딩

두 영상을 합성하려고 할 때 앞서 살펴본 더하기(+) 연산이나 `cv2.add()` 함수만으로는 좋은 결과를 얻을 수 없는 경우가 많습니다. 직접 더하기 연산을 하면 255를 넘는 경우 초과 값만을 가지므로 영상이 거뭇거뭇하게 나타나고 `cv2.add()` 연산을 하면 대부분의 픽셀 값이 255 가까이 몰리는 현상이 일어나서 영상이 하얗게 날아간 것처럼 보입니다. 아래는 이런 현상을 보여주고 있습니다.

```py
'''이미지 단순 합성'''
import cv2
import numpy as np
import matplotlib.pylab as plt

# ---① 연산에 사용할 이미지 읽기
img1 = cv2.imread('./img/wing_wall.jpg')
img2 = cv2.imread('./img/yate.jpg')

# ---② 이미지 덧셈
img3 = img1 + img2  # 더하기 연산
img4 = cv2.add(img1, img2) # OpenCV 함수

imgs = {'img1':img1, 'img2':img2, 'img1+img2':img3, 'cv.add(img1, img2)':img4}

# ---③ 이미지 출력
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2, i + 1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([]); plt.yticks([])

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204094074-c977ec71-416f-4eab-8d93-96048f68d1b5.png">
</p>

실행 결과의 img1+img2는 화소가 고르지 못하고 중간 중간 이상한 색을 띠고 있는 부분이 있는데, 그 부분이 255를 초과한 영역입니다. `cv2.add(img1, img2)` 의 실행 결과는 전체적으로 하얀 픽셀을 많이 가져가므로 좋은 결과로 볼 수 없습니다.

**두 영상을 합성하려면 각 픽셀의 합이 255가 되지 않게 각각의 영상에 가중치를 줘서 계산해야 합니다.** 예를 들어 두 영상이 정확히 절반씩 반영된 결과 영상을 원한다면 각 영상의 픽셀 값에 각각 50%씩 곱해서 새로운 영상을 생성하면 됩니다. 이것을 수식으로 나타내면 다음과 같고 이때 **각 영상에 적용할 가중치를 알파(alpha) 값**이라고 부릅니다. 알파 값을 조정해서 7:3, 6:4, 5:5 등과 같이 배분하는 방식입니다.

$$
g(x) = (1 - \alpha) f_0(x) + \alpha f_1(x)
$$

* $f_0(x)$ : 첫 번째 이미지 픽셀 값
* $f_1(x)$ : 두 번째 이미지 픽셀 값
* $\alpha$ : 가중치(알파)
* $g(x)$ : 합성 결과 픽셀 값

<br>

이 수식대로 NumPy 배열에 직접 연산해도 되지만, OpenCV는 이것을 구현한 함수를 제공합니다.

* `cv2.addWeight(img1, alpha, img2, beta, gamma)`
    * `img1, img2` : 합성할 두 영상
    * `alpha` : img1에 지정할 가중치(알파 값)
    * `beta` : img2에 지정할 가중치, 흔히 (1- alpha) 적용
    * `gamma` : 연산 결과에 가감할 상수, 흔히 0(zero) 적용

아래 코드는 각 영상에 대해서 50%씩 가중치로 앞서 실습한 영상을 다시 합성하고 있습니다.

```py
'''50% 알파 블렌딩'''
import cv2
import numpy as np

alpha = 0.5 # 합성에 사용할 알파 값

#---① 합성에 사용할 영상 읽기
img1 = cv2.imread('./img/wing_wall.jpg')
img2 = cv2.imread('./img/yate.jpg')

# ---② NumPy 배열에 수식을 직접 연산해서 알파 블렌딩 적용
blended = img1 * alpha + img2 * (1-alpha)
blended = blended.astype(np.uint8) # 소수점 발생을 제거하기 위함
cv2.imshow('img1 * alpha + img2 * (1-alpha)', blended)

# ---③ addWeighted() 함수로 알파 블렌딩 적용
dst = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0) 
cv2.imshow('cv2.addWeighted', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204094616-df0e8fa5-d96c-4d39-8514-9449a8ffb744.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204094630-d04834b9-efc1-4520-88fe-a172c0b69925.png">
</p>

위 코드 ②는 앞서 수식으로 나타낸 알파 블렌딩을 NumPy 배열에 직접 적용하였고, 코드 ③은 `cv2.addWeighted()` 함수로 적용해서 같은 결과를 가져오는 것을 보여주고 있습니다.

아래의 코드는 남자의 얼굴과 사자의 얼굴을 알파 블렌딩하는 데 트랙바로 알파 값을 조정할 수 있게 했습니다. 트랙바를 움직여서 알파 값을 조정하면 마치 사람이 서서히 사자로 바뀌는 것처럼 보입니다. **알파 블렌딩은 흔히 페이드-인/아웃(fade-in/out) 기법으로 영상이 전환되는 장면에서 자주 사용**되며, 《구미호》나 《늑대인간》 같은 영화의 **변신 장면에서 얼굴 모핑(face morphing)이라는 기법으로 효과를 내는데, 이 기법을 구성하는 한 가지 기술이기도 합니다.**

```py
'''트랙바로 알파 블렌딩'''
import cv2
import numpy as np

win_name = 'Alpha blending'     # 창 이름
trackbar_name = 'fade'          # 트렉바 이름

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow(win_name, dst)

# ---② 합성 영상 읽기
img1 = cv2.imread('./img/man_face.jpg')
img2 = cv2.imread('./img/lion_face.jpg')

# ---③ 이미지 표시 및 트렉바 붙이기
cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204095271-5faa8841-c355-4b5a-95d8-a269046ec821.png">
</p>

값으로 직접 비교하면 다음과 같습니다.


<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204095259-c51cfc10-2793-4f72-af8e-0753296c6e81.png">
</p>

<br>



## 3. 비트와이즈 연산

**OpenCV는 두 영상의 각 픽셀에 대한 비트와이즈(bitwise, 비트 단위) 연산 기능을 제공**합니다. **비트와이즈 연산은 영상을 합성할 때 특정 영역만 선택하거나 특정 영역만 제외하는 등의 선별적인 연산에 도움이 됩니다.** OpenCV에서 제공하는 비트와이즈 연산 함수는 다음과 같습니다.

* `bitwise_and(img1, img2, mask=None)` : 각 픽셀에 대해 비트와이즈 AND 연산
* `bitwise_or(img1, img2, mask=None)` : 각 픽셀에 대해 비트와이즈 OR 연산
* `bitwise_xor(img1, img2, mask=None)` : 각 픽셀에 대해 비트와이즈 XOR 연산
* `bitwise_not(img1, mask=None)` : 각 픽셀에 대해 비트와이즈 NOT 연산
    * `img1, img2` : 연산 대상 영상, 동일한 shape
    * `mask` : 0이 아닌 픽셀만 연산, 바이너리 이미지

```py
'''비트와이즈 연산'''
import numpy as np, cv2
import matplotlib.pylab as plt

#--① 연산에 사용할 이미지 생성
img1 = np.zeros(( 200,400 ), dtype=np.uint8)
img2 = np.zeros(( 200,400 ), dtype=np.uint8)
img1[:, :200] = 255         # 왼쪽은 흰색(255), 오른쪽은 검정색(0)
img2[100:200, :] = 255      # 위쪽은 검정색(0), 아래쪽은 흰색(255)

#--② 비트와이즈 연산
bitAnd = cv2.bitwise_and(img1, img2)
bitOr = cv2.bitwise_or(img1, img2)
bitXor = cv2.bitwise_xor(img1, img2)
bitNot = cv2.bitwise_not(img1)

#--③ Plot으로 결과 출력
imgs = {'img1':img1, 'img2':img2, 'and':bitAnd, 
          'or':bitOr, 'xor':bitXor, 'not(img1)':bitNot}
for i, (title, img) in enumerate(imgs.items()):
    plt.subplot(3,2,i+1)
    plt.title(title)
    plt.imshow(img, 'gray')
    plt.xticks([]); plt.yticks([])

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204095401-adec7208-f398-4736-9c7d-08b7759413c6.png">
</p>

위 예제의 실행 결과를 보면 이해하기 쉬울 것입니다. img1은 좌우로, img2는 위아래로 0과 255로 나누어 200 × 400 크기의 영상을 생성했습니다. 이 두 영상에 대해서 각각 비트와이즈 연산을 한 결과입니다. `cv2.bitwise_and()` 연산은 두 영상에서 0으로 채워진 부분이 만나는 부분은 모두 0으로 채워졌습니다. `cv2.bitwise_or()` 연산은 두 영상에서 255로 채워진 부분은 모두 255로 채워졌습니다. `cv2.bitwise_xor()` 연산은 두 영상에서 서로 다른 값을 가진 부분은 255로, 서로 같은 값을 가진 부분은 0으로 채워졌습니다. img1에 대한 `cv2.bitwise_not()` 연산은 원래의 반대의 결과를 갖습니다.

다음 코드는 비트와이즈 연산으로 영상의 일부분을 원하는 모양으로 떼내는 예제입니다.

```py
'''bitwise_and 연산으로 마스킹하기'''
import numpy as np, cv2
import matplotlib.pylab as plt

#--① 이미지 읽기
img = cv2.imread('./img/wonyoung.jpg')

#--② 마스크 만들기
mask = np.zeros_like(img)
cv2.circle(mask, (220,270), 150, (255,255,255), -1)
#cv2.circle(대상이미지, (원점x, 원점y), 반지름, (색상), 채우기)

#--③ 마스킹
masked = cv2.bitwise_and(img, mask)

#--④ 결과 출력
cv2.imshow('original', img)
cv2.imshow('mask', mask)
cv2.imshow('masked', masked)
cv2.waitKey()
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204095592-c5f18cda-f833-47b2-90f7-b197fa0131cf.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204095598-2f5c8877-1673-455a-a36c-d6906a4ff4c9.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204095612-0287cc11-09e6-45d0-a0cf-fe00a32faf1b.png">
</p>

위 코드 ②에서 원본 이미지와 동일한 shape의 O(zero)으로 채워진 배열을 만들고 원하는 위치에 (255,255,255)로 채워진 원을 그립니다. 이렇게 생성된 배열은 원을 제외한 나머지 영역은 모두 O(zero)으로 채워져 있고, 원은 모든 비트가 1로 채워져 있는 255입니다. 코드 ③에서는 이 영상과 원본 영상을 `cv2.bitwise_and()` 연산으로 원 이외의 부분을 모두 0으로 채워서 원하는 영역만 떼어낼 수 있습니다.

예제에서는 마스킹하기 위해 코드 ②에서 원본 영상과 똑같은 3채널 배열을 만들었지만, 비트와이즈 연산 함수의 세 번째 인자인 mask를 이용하면 2차원 배열만으로도 가능합니다.

```py
#--② 마스크 만들기
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, (220,270), 150, (255), -1)
#cv2.circle(대상이미지, (원점x, 원점y), 반지름, (색상), 채우기)

#--③ 마스킹
masked = cv2.bitwise_and(img, img, mask=mask)
```

<br>



## 4. 차영상

**영상에서 영상을 빼기 연산하면 두 영상의 차이, 즉 변화를 알 수 있는데,** 이것을 **차영상(image differencing)**이라고 합니다. 심심풀이로 한 번쯤은 해봤을 법한 틀린 그림 찾기 놀이는 차영상으로 손쉽게 답을 찾을 수 있습니다. 놀이뿐만 아니라 산업현장에서 도면의 차이를 찾거나 전자제품의 PCB(Printable Circuit Board) 회로의 오류를 찾는 데도 사용할 수 있고, 카메라로 촬영한 영상에 실시간으로 움직임이 있는지를 알아내는 데도 유용합니다.

**차영상을 구할 때 두 영상을 무턱대고 빼기 연산하면 음수가 나올 수 있으므로 절대 값을 구해야 합니다.** 아래는 OpenCV에서 제공하는 절대 값의 차를 구하는 함수입니다.

* `diff` = cv2.absdiff(img1, img2)
    * `img1, img2` : 입력 영상
    * `diff` : 두 영상의 차의 절대 값 반환

다음 코드는 사람의 눈으로 찾기 힘든 두 도면의 차이를 찾아 표시합니다.

```py
'''차영상으로 도면의 차이 찾아내기'''
import numpy as np, cv2

#--① 연산에 필요한 영상을 읽고 그레이스케일로 변환
img1 = cv2.imread('./img/robot_arm1.jpg')
img2 = cv2.imread('./img/robot_arm2.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#--② 두 영상의 절대값 차 연산
diff = cv2.absdiff(img1_gray, img2_gray)

#--③ 차 영상을 극대화 하기 위해 쓰레시홀드 처리 및 컬러로 변환
_, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
diff_red[:,:,2] = 0

#--④ 두 번째 이미지에 변화 부분 표시
spot = cv2.bitwise_xor(img2, diff_red)

#--⑤ 결과 영상 출력
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('spot', spot)
cv2.waitKey()
cv2.destroyAllWindows()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204096169-77653650-9206-4129-afa8-3179924fa579.png">
</p>

코드 ①은 연산에 필요한 두 영상을 읽어서 그레이 스케일로 변환합니다. 코드 ②에서 그레이 스케일로 변환된 두 영상의 차영상을 구합니다. 그 차이를 극대화해서 표현하기 위해 코드 ③에서는 1보다 큰 값은 모두 255로 바꾸고 색상을 표현하기 위해 컬러 스케일로 바꿉니다. 코드 ④는 원본 이미지의 어느 부분이 변경되었는지 표시해 주기 위해서 `cv2.bitwise_xor()` 연산을 합니다. 원본 이미지는 배경이 흰색이므로 255를 가지고 있고 차영상은 차이가 있는 빨간색 영역을 제외하고는 255이므로 XOR 연산을 하면 서로 다른 영역인 도면의 그림과 빨간색으로 표시된 차영상 부분이 합성됩니다.

<br>



## 5. 이미지 합성과 마스킹

두 개 이상의 영상에서 특정 영역끼리 합성하기 위해서는 전경이 될 영상과 배경이될 영상에서 합성하고자 하는 영역만 떼어내는 작업과 그것을 합하는 작업으로 나눌수 있습니다. 여기서 **원하는 영역만을 떼어내는 데 꼭 필요한 것이 마스크(mask)**입니다. 사람이 좌표를 입력하지 않고 정교한 마스크를 만드는 작업은 결코 쉽지 않습니다. 사실 원하는 영역을 배경에서 떼어내는 작업은 객체 인식과 분리라는 컴퓨터 비전 분야의 정점과도 같다고 볼 수 있습니다.

여기서는 우선 배경이 투명한 알파 채널 영상을 이용해서 영상을 합성해 봅니다. 배경이 투명한 영상은 4개 채널 중 마지막 채널은 배경에 해당하는 영역은 0 값을, 전경에 해당하는 영역은 255 값을 갖습니다. 이것을 이용하면 손쉽게 마스크를 만들 수 있습니다. 마스크를 이용해서 전경과 배경을 오려내는 것은 앞서 살펴본 `cv2.bitwise_and()` 연산을 이용하면 쉽습니다.

```py
'''투명 배경 PNG 파일을 이용한 합성'''
import cv2
import numpy as np

#--① 합성에 사용할 영상 읽기, 전경 영상은 4채널 png 파일
img_fg = cv2.imread('./img/opencv_logo.png', cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread('./img/wonyoung.jpg')

#--② 알파채널을 이용해서 마스크와 역마스크 생성
_, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#--③ 전경 영상 크기로 배경 영상에서 ROI 잘라내기
img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
h, w = img_fg.shape[:2]
roi = img_bg[10:10+h, 10:10+w]

#--④ 마스크 이용해서 오려내기
masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#--⑥ 이미지 합성
added = masked_fg + masked_bg
img_bg[10:10+h, 10:10+w] = added

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('masked_fg', masked_fg)
cv2.imshow('masked_bg', masked_bg)
cv2.imshow('added', added)
cv2.imshow('result', img_bg)
cv2.waitKey()
cv2.destroyAllWindows() 
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097684-9c1c2f13-2ba5-4aba-814e-e339dea8c0b8.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097695-d2464c25-2e03-4fab-8a8a-f42cf7d7a438.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097700-c42650cd-fcde-4510-82d1-3d9180da9f9d.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097710-a1d03e75-4c30-47a6-aaa4-ca5e137818b7.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097716-8a71a06a-7163-4c17-94c2-2fdf9e8f33cf.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097720-9d8e43d0-f80c-4018-b8dc-14b98d188adf.png">
</p>

위 코드는 배경이 투명한 OpenCV 로고 이미지를 사진과 합성하고 있습니다. 로고 이미지의 네 번째 채널이 배경과 전경을 분리할 수 있는 마스크 역할을 해주므로 앞서 설명한 몇 가지 함수의 조합만으로 손쉽게 이미지를 합성할 수 있습니다.

모양에 따라 영역을 떼어내려는 경우도 있지만, 색상에 따라 영역을 떼어내야 하는 경우도 있습니다. 이때는 색을 가지고 마스크를 만들어야 하는데, HSV로 변환하면 원하는 색상 범위의 것만 골라낼 수 있습니다. OpenCV는 특정 범위에 속하는지를 판단할 수 있는 함수를 아래와 같이 제공합니다. 이것을 이용하면 특정 범위 값을 만족하는 마스크를 만들기 쉽습니다.

* `dst = cv2.inRange(img, from, to)` : 범위에 속하지 않은 픽셀 판단
    * `img` : 입력 영상
    * `from` : 범위의 시작 배열
    * `to` : 범위의 끝 배열
    * `dst` : ing가 from ~ to에 포함되면 255, 아니면 0을 픽셀 값으로 하는 배열

다음 코드는 컬러 큐브에서 색상별로 추출하는 예제입니다.

```py
'''HSV 색상으로 마스킹'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 큐브 영상 읽어서 HSV로 변환
img = cv2.imread("./img/cube.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#--② 색상별 영역 지정
blue1 = np.array([90, 50, 50])
blue2 = np.array([120, 255, 255])
green1 = np.array([45, 50, 50])
green2 = np.array([75, 255, 255])
red1 = np.array([0, 50, 50])
red2 = np.array([15, 255, 255])
red3 = np.array([165, 50, 50])
red4 = np.array([180, 255, 255])
yellow1 = np.array([20, 50, 50])
yellow2 = np.array([35, 255, 255])

# --③ 색상에 따른 마스크 생성
mask_blue = cv2.inRange(hsv, blue1, blue2)
mask_green = cv2.inRange(hsv, green1, green2)
mask_red = cv2.inRange(hsv, red1, red2)
mask_red2 = cv2.inRange(hsv, red3, red4)
mask_yellow = cv2.inRange(hsv, yellow1, yellow2)

#--④ 색상별 마스크로 색상만 추출
res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
res_green = cv2.bitwise_and(img, img, mask=mask_green)
res_red1 = cv2.bitwise_and(img, img, mask=mask_red)
res_red2 = cv2.bitwise_and(img, img, mask=mask_red2)
res_red = cv2.bitwise_or(res_red1, res_red2)
res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

#--⑤ 결과 출력
imgs = {'original': img, 'blue':res_blue, 'green':res_green, 
                            'red':res_red, 'yellow':res_yellow}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2, 3, i+1)
    plt.title(k)
    plt.imshow(v[:,:,::-1])
    plt.xticks([]); plt.yticks([])

plt.show()
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204097865-66450709-a053-4e9a-8dd0-b918021b6b81.png">
</p>

코드 ②에서 지정한 색상별 영역은 HSV의 각 색상별 영역에서 설명한 것을 근거로 작성하였습니다. 빨강은 180을 기점으로 둘로 나뉘어(0~15, 165~180) 있으므로 마스크 생성과 색상 추출에도 두 번씩 사용했습니다. 코드 ③에서 `cv2.inRange()` 함수를 호출해서 각 색상 범위별 마스크를 만듭니다. 이 함수는 첫 번째 인자의 영상에서 두 번째와 세 번째 인자의 배열 구간에 포함되면 해당 픽셀의 값으로 255를 할당하고 그렇지 않으면 0을 할당합니다. 그래서 이 함수의 반환 결과는 바이너리 스케일이 되어 코드 ④의 `cv2․bitwise_and()` 함수의 mask로 사용하기 적합합니다.

이와 같이 **색상을 이용한 마스크를 이용하는 것이 크로마 키(chroma key)의 원리**입니다. 일기예보나 영화를 촬영할 때 **초록색 또는 파란색 배경을 두고 찍어서 나중에 원하는 배경과 합성하는 것을 크로마 키잉(chroma keying)**이라고 하고 **그 초록색 배경을 크로마 키**라고 합니다.


<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204100127-45bf5c84-9bc2-4438-9e97-f1946b195d86.jpg">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204100151-2aaa9362-682e-43d9-a07c-b78c0fd23b38.jpg">
</p>

다음 코드는 크로마 키를 배경으로 한 영상에서 크로마 키 색상으로 마스크를 만들어 합성하는 예제로, 위 두 사진을 이용하겠습니다.

```py
'''크로마키 마스킹과 합성'''
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 크로마키 배경 영상과 합성할 배경 영상 읽기
img1 = cv2.imread('./img/man_chromakey.jpg')
img2 = cv2.imread('./img/street.jpg')

#--② ROI 선택을 위한 좌표 계산
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]
x = (width2 - width1)//2
y = height2 - height1
w = x + width1
h = y + height1
print('height1 : ', height1, 'width1 : ', width1)
print('height2 : ', height2, 'width2 : ', width2)
print('x : ', x)
print('y : ', y)
print('w : ', w)
print('h : ', h)

#--③ 크로마키 배경 영상에서 크로마키 영역을 10픽셀 정도로 지정
chromakey = img1[:10, :10, :]
offset = 20

#--④ 크로마키 영역과 영상 전체를 HSV로 변경
hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

#--⑤ 크로마키 영역의 H값에서 offset 만큼 여유를 두어서 범위 지정
# offset 값은 여러차례 시도 후 결정
#chroma_h = hsv_chroma[0]
chroma_h = hsv_chroma[:,:,0]
lower = np.array([chroma_h.min()-offset, 100, 100])
upper = np.array([chroma_h.max()+offset, 255, 255])

#--⑥ 마스크 생성 및 마스킹 후 합성
mask = cv2.inRange(hsv_img, lower, upper)
mask_inv = cv2.bitwise_not(mask)
roi = img2[y:h, x:w]
cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('roi', roi)

fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
bg = cv2.bitwise_and(roi, roi, mask=mask)
img2[y:h, x:w] = fg + bg

#--⑦ 결과 출력
cv2.imshow('fg', fg)
cv2.imshow('bg', bg)

cv2.imshow('chromakey', img1)
cv2.imshow('added', img2)
cv2.waitKey()
cv2.destroyAllWindows()

[output]
height1 :  400 width1 :  314
height2 :  426 width2 :  640
x :  163
y :  26
w :  477
h :  426
```

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204099678-7bf9d60a-249f-459f-81c9-fa04a0448a6b.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204099681-aa2b3260-2253-4868-a8e6-3f64e44be29f.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204099915-1fc133e6-a658-44cb-b007-0fe3f73d27c3.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204099694-552b9bf0-2694-4b7c-af06-523e334c7ee0.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204099702-ab67f5e5-b587-4c02-9628-a318893bc7ef.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204098698-ed0da3a3-d63a-42eb-9ef3-8a0373715cb6.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204098706-9fe360ee-10dc-42ec-bfbe-8fbf2e244866.png">
</p>

왼쪽에 한 남자가 크로마 키를 배경으로 찍은 사진을 어느 거리를 찍은사진과 합성한 것입니다. 코드 ②에서는 남자가 서 있는 왼쪽 끝 배경 10 × 10 픽셀 영역을 크로마 키가 있는 영역으로 어림잡아 지정했습니다. 이 영역의 색상 값 중에 가장 큰 값과 가장 작은 값을 범위로 지정해서 `cv2.inRange()` 함수를 사용하면 배경만 제거할 수 있습니다. 코드 ④에서는 앞서 어림잡아 선택한 영역의 색상 값보다 더 넓은 영역의 색상을 선택할 수 있도록 offset 만큼 가감하게 했고 그 수치는 결과를 확인하면서 경험적으로 얻어야 합니다. 크로마 키의 색상 값도 화면 전체적으로는 조금씩 다를 수 있기 때문입니다. S와 V 값의 선택 범위도 마찬가지입니다. 나머지 마스킹과 합성 작업은 이전에 했던 것과 크게 다르지 않습니다.

이렇게 **영상 합성에는 대부분 알파 블렌딩 또는 마스킹이 필요합니다.** 하지만, 이런 작업은 블렌딩을 위한 적절한 알파 값 선택과 마스킹을 위한 모양의 좌표나 색상값 선택에 많은 노력과 시간이 필요합니다. OpenCV는 3 버전에서 재미있는 함수를 추가했는데, 알아서 두 영상의 특징을 살려 합성하는 기능입니다. 이 함수의 설명은 아래와 같습니다.

* `dst = cv2.seamlessClone(src, dst, mask, coords, flags[, output])`
    * `src` : 입력 영상, 일반적으로 전경
    * `dst` : 대상 영상, 일반적으로 배경
    * `mask` : 마스크, src에서 합성하고자 하는 영역은 255, 나머지는 0
    * `coodrs` : src가 놓여지기 원하는 dst의 좌표(중앙)
    * `flags` : 합성 방식
        * `cv2.NORMAL_CLONE` : 입력 원본 유지
        * `cv2.MIXED_CLONE` : 입력과 대상을 혼합
    * `output` : 합성 결과
    * `dst` : 합성 결과

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204100218-c0c964bf-8fd3-40ae-b9b3-5c89786b47b2.jpg">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204100232-743464be-dad3-4e6d-9c06-8ae8777cd404.jpg">
</p>

위 사진을 이용하여, `cv2.SeamlessClone()` 함수로 사진을 합성해서 손에 꽃 문신을 한 것처럼 만들어 보겠습니다.

```py
'''SeamlessClone으로 합성'''
import cv2
import numpy as np
import matplotlib.pylab as plt
 
#--① 합성 대상 영상 읽기
img1 = cv2.imread("./img/drawing.jpg")
img2 = cv2.imread("./img/my_hand.jpg")

#--② 마스크 생성, 합성할 이미지 전체 영역을 255로 셋팅
mask = np.full_like(img1, 255)
 
#--③ 합성 대상 좌표 계산(img2의 중앙)
height, width = img2.shape[:2]
center = (width//2, height//2)
 
#--④ seamlessClone 으로 합성 
normal = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)
mixed = cv2.seamlessClone(img1, img2, mask, center, cv2.MIXED_CLONE)

#--⑤ 결과 출력
cv2.imshow('normal', normal)
cv2.imshow('mixed', mixed)
cv2.waitKey()
cv2.destroyAllWindows()
```

위 코드 ④가 이 예제의 핵심적인 코드입니다. img1을 img2에다가 mask에 지정된 영역만큼 center 좌표에 합성합니다. 이때 mask는 img1의 전체 영역을 255채워서 해당 영역 전부가 합성의 대상임을 표현합니다. 가급적이면 합성하려는 영역을 제외하고 0으로 채우는 것이 더 좋은 결과를 보여주지만 이번 예제에서는 일부러 대충해 보았습니다. 결과를 보면 함수의 마지막 인자 플래그가 `cv2.NORMAL_CLONE` 인 경우 꽃 그림이 선명하긴 하지만, 주변의 피부가 뭉개진 듯한 결과를 보입니다. 반면에, `cv2.MIXED_CLONE` 을 사용한 경우에는 감쪽같이 두 영상의 특징을 살려서 표현하고 있습니다. 이 함수는 이미지 합성에 꼭 필요한 알파 값이나 마스크에 대해 신경 쓰지 않아도 되서 무척 편리합니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204100364-bf7e5b40-eb85-4779-88b5-a089345bd592.png">
</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/204100370-3d72c7b2-6ed5-4646-b551-ca52682bf1c3.png">
</p>




