---
layout: post
title: OpenCV를 이용한 이미지와 비디오 입출력
category: OpenCV
tag: OpenCV
---

[![Hits](https://hits.sh/museonghwang.github.io.svg?view=today-total&style=for-the-badge&label=Visitors&color=007ec6)](https://hits.sh/museonghwang.github.io/)

<br>

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.

<br>





# 이미지와 비디오 입출력

OpenCV를 이용한 대부분의 작업은 파일로 된 이미지를 읽어서 적절한 연산을 적용하고 그 결과를 화면에 표시하거나 다른 파일로 저장하는 것입니다. 여기서는 이미지 파일을 읽고 화면에 표시하고 저장하는 방법을 중점적으로 살펴보겠습니다.



## 1. 이미지 읽기

OpenCV를 사용해서 이미지를 읽고 화면에 표시하는 가장 간단한 코드는 아래와 같습니다.

```py
'''이미지 파일을 화면에 표시'''
import cv2

img_file = "./img/wonyoung.jpg" # 표시할 이미지 경로            ---①
img = cv2.imread(img_file)      # 이미지를 읽어서 img 변수에 할당 ---②

if img is not None:
    cv2.imshow('IMG', img)      # 읽은 이미지를 화면에 표시  ---③
    cv2.waitKey()               # 키가 입력될 때 까지 대기  ---④
    cv2.destroyAllWindows()     # 창 모두 닫기           ---⑤
else:
    print('No image file.')
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202856815-75039f19-03de-423e-afd8-fa624ecc3d6c.png">
</p>

코드 ①의 경로에 표시할 이미지 파일이 저장되어 있어야 합니다. 이 파일을 코드 ②에서 `cv2.imread()` 함수로 읽어들입니다. 이 함수가 반환하는 타입은 NumPy 배열입니다. 이 반환 값이 정상인지 아닌지 확인하고 나서 코드 ③에서 `cv2․imshow()` 함수를 써서 화면에 표시합니다. 이미지와 함께 전달한 문자열 'IMG'는 창의 제목줄에 나타납니다.

만약 코드가 코드 ③까지만 작성되어 있다면 더 이상 실행할 코드가 없어서 프로그램은 바로 종료될 것입니다. 그렇게 되면 사진을 표시한 이 창은 아주 짧은 시간 동안만 나타나 우리 눈으로는 볼 수 없게 됩니다. 그래서 코드 ④가 필요합니다. `cv2.waitKey()` 함수는 키보드의 입력이 있을 때까지 프로그램을 기다리게 합니다. 키가 입력되면 코드는 코드 ⑤의 `cv2.destroyAllWindows()` 함수에 의해서 표시한 창을 모두 닫고 나서 프로그램을 종료합니다.

위에서 사용한 함수는 다음과 같습니다.
* `img = cv2.imread(file_name [, mode_flag])` : 파일로부터 이미지 읽기
    * `file_name` : 이미지 경로, 문자열
    * `mode_flag=cv2.IMREAD_COLOR` : 읽기 모드 지정
        * `cv2.IMREAD_COLOR` : 컬러(BGR) 스케일로 읽기, 기본 값
        * `cv2.IMREAD_UNCHANGED` : 파일 그대로 읽기
        * `cv2.IMREAD_GRAYSCALE` : 그레이(흑백) 스케일로 읽기
    * `img` : 읽은 이미지, NumPy 배열
* `cv2.imshow(title, img)` : 이미지를 화면에 표시
    * `title` : 창 제목, 문자열
    * `img` : 표시할 이미지, NumPy 배열
* `key = cv2.waitKey([delay])` : 키보드 입력 대기
    * `delay=0` : 키보드 입력을 대기할 시간(ms), 0: 무한대(기본 값)
    * `key` : 사용자가 입력한 키 값, 정수
        * `-1` : 대기시간 동안 키 입력 없음

`cv2.imread()` 함수는 파일로부터 이미지를 읽을 때 모드를 지정할 수 있습니다. 별도로 모드를 지정하지 않으면 3개 채널(B, G, R)로 구성된 컬러 스케일로 읽어들이지만, 필요에 따라 그레이 스케일 또는 파일에 저장된 스케일 그대로 읽을 수 있습니다.

```py
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
```

위의 코드와 같이 읽기 모드를 그레이 스케일로 지정하면 원래의 파일이 컬러 이미지일지라도 그레이 스케일로 읽습니다. 물론 그레이 이미지 파일을 `cv2.IMREAD_COLOR` 옵션을 지정해서 읽는다고 컬러 이미지로 읽어올 수 있는 것은 아닙니다.

```py
'''이미지 파일을 그레이 스케일로 화면에 표시'''
import cv2

img_file = "./img/wonyoung.jpg" 
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) # 그레이 스케일로 읽기

if img is not None:
    cv2.imshow('IMG', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No image file.')
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202857369-e3e624cf-08ba-4d68-a11f-4db7c935ef77.png">
</p>




## 2. 이미지 저장하기

OpenCV로 읽어들인 이미지를 다시 파일로 저장하는 함수는 `cv2.imwrite()` 입니다.
* `cv2.imwrite(file_path, img)` : 이미지를 파일에 저장
    * `file_path` : 저장할 파일 경로 이름, 문자열
    * `img` : 저장할 영상, NumPy 배열

다음 코드는 컬러 이미지 파일을 그레이 스케일로 읽어들여서 파일로 저장하는 예제입니다. 탐색기나 파인더 등과 같은 파일 관리자로 해당 경로를 살펴보면 그레이 스케일로 바뀐 새로운 파일이 저장된 것을 확인할 수 있습니다. 저장하는 이미지의 파일 포맷은 지정한 파일 이름의 확장자에 따라서 알아서 바뀝니다.

```py
'''컬러 이미지를 그레이 스케일로 저장'''
import cv2

img_file = './img/wonyoung.jpg'
save_file = './img/wonyoung_gray.jpg'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
cv2.imshow(img_file, img)
cv2.imwrite(save_file, img) # 파일로 저장, 포맷은 확장에 따름
cv2.waitKey()
cv2.destroyAllWindows()
```



## 3. 동영상 및 카메라 프레임 읽기

OpenCV는 동영상 파일이나 컴퓨터에 연결한 카메라 장치로부터 연속된 이미지 프레임을 읽을 수 있는 API를 제공합니다.

다음은 동영상 파일이나 연속된 이미지 프레임을 읽을 수 있는 API의 주요 내용입니다.
* `cap = cv2․VideoCapture(file_path 또는 index)` : 비디오 캡처 객체 생성자
    * `file_path` : 동영상 파일 경로
    * `index` : 카메라 장치 번호, 0부터 순차적으로 증가(0, 1, 2, ...)
    * `cap` : VideoCapture 객체
* `ret = cap.isOpened()` : 객체 초기화 확인
    * `ret` : 초기화 여부, True/False
* `ret, img = cap.read()` : 영상 프레임 읽기
    * `ret` : 프레임 읽기 성공 또는 실패 여부, True/False
    * `img` : 프레임 이미지, NumPy 배열 또는 None
* `cap.set(id, value)` : 프로퍼티 변경
* `cap.get(id)` : 프로퍼티 확인
* `cap.release()` : 캡처 자원 반납

동영상 파일이나 컴퓨터에 연결한 카메라 장치로부터 영상 프레임을 읽기 위해서는 `cv2.VideoCapture()` 생성자 함수를 사용하여 객체를 생성해야 합니다. 이 함수에 동영상 파일 경로 이름을 전달하면 동영상 파일에 저장된 프레임을 읽을 수 있고, 카메라 장치 번호를 전달하면 카메라로 촬영하는 프레임을 읽을 수 있습니다.

객체를 생성하고 나면 `isOpened()` 함수로 파일이나 카메라 장치에 제대로 연결되었는지 확인할 수 있고, 연결이 잘 되었다면 `read()` 함수로 다음 프레임을 읽을 수 있습니다. `read()` 함수는 Boolean과 NumPy 배열 객체를 쌍으로 갖는 튜플 (ret, img) 객체를 반환하는데, 다음 프레임을 제대로 읽었는지에 따라 `ret` 값이 정해집니다. 만약 `ret` 값이 `True` 이면 다음 프레임 읽기에 성공한 것이고, `img` 를 꺼내서 사용하면 됩니다. 만약 `ret` 값이 `False` 이면 다음 프레임 읽기에 실패한 것이고, 튜플의 나머지 값인 `img` 는 `None` 입니다. 다음 프레임 읽기에 실패하는 경우는 파일이나 장치에 문제가 있거나 파일의 끝에 도달했을 경우입니다.

비디오 캡처 객체의 `set()`, `get()` 함수를 이용하면 여러 가지 속성을 얻거나 지정할 수 있으며, 프로그램을 종료하기 전에 `release()` 함수를 호출해서 자원을 반납해야 합니다.



## 4. 동영상 파일 읽기

다음은 동영상 파일을 읽기 위한 간단한 코드입니다.

```py
'''동영상 파일 재생'''
import cv2

video_file = "./img/big_buck.avi"   # 동영상 파일 경로

cap = cv2.VideoCapture(video_file)  # 동영상 캡쳐 객체 생성 ---①
if cap.isOpened():                  # 캡쳐 객체 초기화 확인
    while True:
        ret, img = cap.read()       # 다음 프레임 읽기 ---②
        if ret:                     # 프레임 읽기 정상
            cv2.imshow(video_file, img) # 화면에 표시 ---③
            cv2.waitKey(25)             # 25ms 지연(40fps로 가정) ---④
        else:                       # 다음 프레임 읽을 수 없슴,
            break                   # 재생 완료
else:
    print("can't open video.")      # 캡쳐 객체 초기화 실패

cap.release()                       # 캡쳐 자원 반납
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202857965-a920fe8d-b4d5-4f74-8d11-90fc8cd9bc82.png">
</p>

코드 ①에서는 `cv2.VideoCapture()` 함수에 동영상 파일 경로를 전달해서 캡처 객체 `cap` 을 생성합니다. 캡처 객체가 정상적으로 지정한 파일로 초기화되면 `cap.isOpened()` 함수는 `True` 를 반환합니다. 연속해서 파일의 프레임을 읽어오기 위해서 무한 루프를 돌리면서 `cap.read()` 를 호출하는데, 이 함수는 정상적인 프레임 읽기가 되었는지를 확인할 수 있는 불(boolean) 변수와 한 개의 프레임 이미지를 표현한 NumPy 배열 객체를 쌍으로 갖는 튜플 객체를 반환합니다. 그 다음 프레임 이미지를 화면에 표시하는 것은 이전의 코드와 거의 비슷합니다.

코드 ④에서 `cv2.waitKey(25)` 가 필요한 이유는 각 프레임을 화면에 표시하는 시간이 너무 빠르면 우리 눈으로 볼 수 없기 때문입니다. 이때 지연 시간은 동영상의 *FPS(Frames Per Second, 초당 프레임 수)* 에 맞게 조정해서 적절한 속도로 영상을 재생하게 해야 합니다.



### FPS와 지연 시간 구하기

동영상 파일의 정확한 FPS를 쉽게 얻는 방법은 곰플레이어, 다음팟플레이어, VLC 등과 같은 무료 동영상 플레이어에서 속성 값을 확인하는 것입니다. FPS를 대충 추정하거나 다른플레이어로 구했다면 이에 맞는 지연 시간을 구해야 할 것입니다. FPS에 맞는 지연 시간을 구하는 공식은 1초에 몇 개의 사진이 들어가야 하는가를 구하는 것으로 다음과 같습니다.

$$지연시간 = 1000 : fps$$

1,000으로 계산하는 이유는 1초를 밀리초(ms) 단위로 환산해서 제공해야 하기 때문입니다. FPS를 40으로 가정해서 대입한 결과는 다음과 같습니다.

$$25 = 1000 / 40$$



## 5. 카메라(웹캠) 프레임 읽기

카메라로 프레임을 읽기 위해서는 `cv2.VideoCapture()` 함수에 동영상 파일 경로 대신에 카메라 장치 인덱스 번호를 정수로 지정해 주면 됩니다. 카메라 장치 인덱스 번호는 0부터 시작해서 1씩 증가합니다. 만약 카메라가 하나만 연결되어 있으면 당연히 0번 인덱스를 사용하면 됩니다. 이 부분을 제외하고는 나머지 코드는 동영상 파일을 읽는 것과 거의 똑같습니다.

```py
'''카메라 프레임 읽기'''
import cv2

cap = cv2.VideoCapture(0)               # 0번 카메라 장치 연결 ---①
if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()           # 다음 프레임 읽기
        if ret:
            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            if cv2.waitKey(1) != -1:    # 1ms 동안 키 입력 대기 ---②
                break                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()                           # 자원 반납
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202858736-ec2a2d30-2e03-43f4-8b7b-ff150c611a53.png">
</p>

코드 ①에서는 0번 카메라 장치에서 촬영한 프레임을 읽어서 화면에 표시합니다. 동영상 파일과는 다르게 카메라로부터 프레임을 읽는 경우 파일의 끝이 정해져 있지 않으므로 무한루프를 빠져 나올 조건이 없습니다. 그래서 코드 ②에서 사용자가 아무 키나 누르면 빠져 나오게 했습니다. 따라서 이 프로그램을 종료하려면 키보드의 아무 키나 누르면 됩니다. `cv2.waitKey()` 함수는 지정한 대기 시간 동안 키 입력이 없으면 -1을 반환합니다. 반환된 값이 -1이 아니면 당연히 아무 키나 입력되었다는 뜻입니다.



## 6. 카메라 비디오 속성 제어

캡처 객체에는 영상 또는 카메라의 여러 가지 속성을 확인하고 설정할 수 있는 `get(id)`, `set(id, value)` 함수를 제공합니다. 속성을 나타내는 아이디는 `cv2.CAP_PROP_` 으로 시작하는 상수로 정의되어 있습니다.
* 속성 ID : `cv2.CAP_PROP_` 로 시작하는 상수
    * `cv2.CAP_PROP_FRAME_WIDTH` : 프레임 폭
    * `cv2.CAP_PROP_FRAME_HEIGHT` : 프레임 높이
    * `cv2.CAP_PROP_FPS` : 초당 프레임 수
    * `cv2.CAP_PROP_POS_MSEC` : 동영상 파일의 프레임 위치(ms)
    * `cv2.CAP_PROP_POS_AVI_RATIO` : 동영상 파일의 상대위치(0: 시작, 1: 끝)
    * `cv2.CAP_PROP_FOURCC` : 동영상 파일 코덱 문자
    * `cv2.CAP_PROP_AUTOFOCUS` : 카메라 자동 초점 조절
    * `cv2.CAP_PROP_ZOOM` : 카메라 줌

각 속성 아이디를 `get()` 에 전달하면 해당 속성의 값을 구할 수 있고, `set()` 함수에 아이디와 값을 함께 전달하면 값을 지정할 수 있습니다.

앞서 동영상 파일을 재생하는 코드에서는 적절한 FPS에 따라 지연 시간을 설정해야하지만, FPS를 대충 짐작하거나 별도의 플레이어를 활용해서 알아내야 했습니다. 비디오 속성 중에 FPS를 구하는 상수는 `cv2.CAP_PROP_FPS` 이고 이것으로 동영상의 FPS를 구하고 다음과 같이 적절한 지연 시간을 계산해서 지정할 수 있습니다.

```py
fps = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수 구하기
delay = int(1000/fps)           # 지연 시간 구하기
```

`cv2.waitKey()` 함수에 전달하는 지연 시간은 밀리초(1/1000초) 단위이고 정수만 전달할 수 있으므로 1초를 1000으로 환산해서 계산한 뒤 정수형으로 바꿉니다. FPS에 맞는 지연 시간을 지정해서 완성한 코드는 다음과 같습니다.

```py
'''FPS를 지정해서 동영상 재생'''
import cv2

video_file = "./img/big_buck.avi"   # 동영상 파일 경로

cap = cv2.VideoCapture(video_file)  # 동영상 캡쳐 객체 생성
if cap.isOpened():                  # 캡쳐 객체 초기화 확인
    fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
    delay = int(1000/fps)
    print("FPS: %f, Delay: %dms" %(fps, delay))

    while True:
        ret, img = cap.read()       # 다음 프레임 읽기
        if ret:                     # 프레임 읽기 정상
            cv2.imshow(video_file, img) # 화면에 표시
            cv2.waitKey(delay)          # fps에 맞게 시간 지연
        else:
            break                       # 다음 프레임 읽을 수 없음, 재생 완료
else:
    print("can't open video.")      # 캡쳐 객체 초기화 실패

cap.release()                       # 캡쳐 자원 반납
cv2.destroyAllWindows()

[output]
FPS: 24.000000, Delay: 41ms
```

아쉽게도 FPS 속성을 카메라 장치로부터 읽을 때는 대부분 정상적인 값을 가져오지 못합니다.

다른 예시로 카메라로부터 읽은 영상이 너무 고화질인 경우 픽셀 수가 많아 연산하는 데 시간이 많이 걸리는 경우가 있습니다. 이때 프레임의 폭과 높이를 제어해서 픽셀 수를 줄일 수 있습니다. 프레임의 폭과 높이 속성 아이디 상수는 `cv2.CAP_PROP_FRAME_WIDTH` 와 `cv2.CAP_PROP_FRAME_HEIGHT` 입니다. 카메라 기본 영상 프레임의 폭과 높이를 구해서 출력하고 새로운 크기를 지정하는 코드는 다음과 같습니다.

```py
'''카메라 프레임 크기 설정'''
import cv2

cap = cv2.VideoCapture(0)                   # 카메라 0번 장치 연결

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # 프레임 폭 값 구하기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 프레임 높이 값 구하기
print("Original width: %d, height:%d" % (width, height) ) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)      # 프레임 폭을 320으로 설정 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)     # 프레임 높이를 240으로 설정

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # 재지정한 프레임 폭 값 구하기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 재지정한 프레임 폭 값 구하기
print("Resized width: %d, height:%d" % (width, height) )

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('camera', img)
            if cv2.waitKey(1) != -1:
                break
        else:
            print('no frame!')
            break
else:
    print("can't open camera!")

cap.release()
cv2.destroyAllWindows()

[output]
Original width: 1280, height:720
Resized width: 640, height:480
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202859330-bc8277d2-404b-4d5e-bd33-5de3a1add916.png">
</p>

파이썬 콘솔에는 위와 같이 원래의 프레임 크기와 새로 지정한 프레임 크기가 출력됩니다. 아쉽게도 카메라가 아닌 동영상 파일에 프레임 크기를 재지정하는 것은 적용되지 않습니다.



## 7. 비디오 파일 저장하기

카메라나 동영상 파일을 재생하는 도중 특정한 프레임만 이미지로 저장하거나 특정 구간을 동영상 파일로 저장할 수도 있습니다. 한 개의 특정 프레임만 파일로 저장하는 방법은 `cv2.imwirte()` 함수를 그대로 사용하면 됩니다.

다음 예제는 카메라로부터 프레임을 표시하다가 아무 키나 누르면 해당 프레임을 파일로 저장하는 코드입니다. 흔히 디지털 카메라로 사진을 찍는 것과 같다고 할 수 있습니다.

```py
'''카메라로 사진 찍기'''
import cv2

cap = cv2.VideoCapture(0)                       # 0번 카메라 연결
if cap.isOpened() :
    while True:
        ret, frame = cap.read()                 # 카메라 프레임 읽기
        if ret:
            cv2.imshow('camera', frame)         # 프레임 화면에 표시
            if cv2.waitKey(1) != -1:            # 아무 키나 누르면
                cv2.imwrite('photo.jpg', frame) # 프레임을 'photo.jpg'에 저장
                break
        else:
            print('no frame!')
            break
else:
    print('no camera!')

cap.release()
cv2.destroyAllWindows()
```

위 코드를 실행하면 카메라로부터 촬영한 영상이 화면에 나오는데, 카메라를 보고자 세를 취하면서 키보드의 아무 키나 누르면 코드를 실행한 디렉터리에 photo.jpg로 사진이 저장됩니다.

하나의 프레임이 아닌 여러 프레임을 동영상으로 저장하려고 할 때는 `cv2.VideoWriter()` 라는 API가 필요합니다.
* `writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))` : 비디오 저장 클래스 생성자 함수
    * `file_path` : 비디오 파일 저장 경로
    * `fourcc` : 비디오 인코딩 형식 4글자
    * `fps` : 초당 프레임 수
    * `(width, height)` : 프레임 폭과 프레임 높이
    * `writer` : 생성된 비디오 저장 객체
* `writer.write(frame)` : 프레임 저장
    * `frame` : 저장할 프레임, NumPy 배열
* `writer.set(id, value)` : 프로퍼티 변경
* `writer.get(id)` : 프로퍼티 확인
* `ret = writer.fourcc(c1, c2, c3, c4)` : fourcc 코드 생성
    * `c1, c2, c3, c4` : 인코딩 형식 4글자, 'MJPG', 'DIVX' 등
    * `ret` : fourcc 코드
* `cv2.VideoWriter_fourcc(c1, c2, c3, c4)` : `cv2.VideoWriter.fourcc()` 와 동일

`cv2․VideoWriter()` 생성자 함수에 저장할 파일 이름과 인코딩 포맷 문자, fps, 프레임 크기를 지정해서 객체를 생성하고 `write()` 함수로 프레임을 파일에 저장하면 됩니다.

`cv2.VideoWriter_fourcc()` 함수는 4개의 인코딩 포맷 문자를 전달하면 코드 값을 생성해 내는 함수로, ‘DIVX'를 예로 들면 다음 두 코드는 그 결과가 똑같습니다.

```py
fourcc = cv2.VideoWriter_foucc(*"DIVX")

# or
fourcc = ord('D') + (ord('I') << 8) + (ord('V') << 16) + (ord('X') << 24)
```

결국 4개의 문자를 한 문자당 8비트씩을 사용해서 각 자릿수에 맞게 표현한 것입니다.

```py
import cv2

cap = cv2.VideoCapture(0)   # 0번 카메라 연결

if cap.isOpened:
    file_path = './record.avi'  # 저장할 파일 경로 이름 ---①
    fps = 30.0                  # FPS, 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')    # 인코딩 포맷 문자
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))                    # 프레임 크기
    out = cv2.VideoWriter(file_path, fourcc, fps, size) # VideoWriter 객체 생성
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera-recording',frame)
            out.write(frame)                        # 파일 저장
            if cv2.waitKey(int(1000/fps)) != -1: 
                break
        else:
            print("no frame!")
            break
    out.release()                                   # 파일 닫기
else:
    print("can't open camera!")

cap.release()
cv2.destroyAllWindows()
```

위 코드를 실행하면 카메라 영상이 화면에 나타나고 코드 ①에서 지정한 경로에 동영상이 녹화되어 저장되기 시작하고 키보드의 아무 키나 누르면 종료됩니다. 탐색기나 파인더와 같은 파일 관리자로 코드 ①에서 지정한 경로를 살펴보면 동영상이 저장된 것을 확인할 수 있습니다.




