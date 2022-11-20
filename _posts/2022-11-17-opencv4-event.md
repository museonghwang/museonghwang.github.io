---
layout: post
title: OpenCV 창 관리 및 이벤트 처리
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

file_path = './img/girl.jpg'
img = cv2.imread(file_path)                             # 이미지를 기본 값으로 읽기
img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 그레이 스케일로 읽기

cv2.namedWindow('origin')                   # origin 이름으로 창 생성
cv2.namedWindow('gray', cv2.WINDOW_NORMAL)  # gray 이름으로 창 생성
cv2.imshow('origin', img)                   # origin 창에 이미지 표시
cv2.imshow('gray', img_gray)                # gray 창에 이미지 표시

cv2.moveWindow('origin', 0, 0)              # 창 위치 변경
cv2.moveWindow('gray', 100, 100)            # 창 위치 변경

cv2.waitKey(0)                              # 아무키나 누르면
cv2.resizeWindow('origin', 200, 200)        # 창 크기 변경 (변경 안됨)
cv2.resizeWindow('gray', 100, 100)          # 창 크기 변경 (변경 됨))

cv2.waitKey(0)                              # 아무키나 누르면
cv2.destroyWindow("gray")                   # gray 창 닫기

cv2.waitKey(0)                              # 아무키나 누르면
cv2.destroyAllWindows()                     # 모든 창 닫기
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202864757-e925ff30-45bc-4645-8a47-819a5f77866c.png">
</p>

위 예제는 최초에 'origin'과 'gray'라는 2개의 창을 띄워서 원본 이미지와 그레이 스케일 이미지를 각각 보여주는데, 이때 'origin' 창은 `cv2.WINDOW_AUTOSIZE` 옵션으로 열었고, 'gray' 창은 `cv2.WINDOW_NORMAL` 옵션으로 열었습니다.

화면을 표시한 다음 `cv2.moveWindow()` 함수로 각각의 창을 모니터 좌측 상단으로 이동시킨 다음 아무 키나 누르면 `cv2.resizeWindow()` 함수로 창의 크기를 변경합니다. 이때, 'origin' 창은 `cv2.WINDOW_AUTOSIZE` 로 창을 열었으므로 창의 크기는 변경되지 않고, `cv2.WINDOW_NORMAL` 옵션을 사용한 'gray' 창은 창의 크기가 변경됩니다. 사용자의 마우스를 이용해서 창의 크기를 변경하는 것도 같습니다.

창의 크기가 변경되고 나서 다시 한번 아무 키나 누르면 'gray' 창만 닫히고 다시한번 아무 키나 누르면 나머지 'origin' 창도 닫힙니다.

이와 같이 OpenCV에서 제공하는 창과 관련한 API는 창을 열 때 사용한 이름을 기반으로 연결되는 것이 특징입니다.

<br>





# 2. 이벤트 처리

키보드와 마우스 입력 방법에 대해 알아봅니다.



## 1. 키보드 이벤트

`cv2.waitKey(delay)` 함수를 쓰면 키보드의 입력을 알아낼 수 있습니다. 이 함수는 delay 인자에 밀리초(ms, 0.001초) 단위로 숫자를 전달하면 해당 시간 동안 프로그램을 멈추고 대기하다가 키보드의 눌린 키에 대응하는 코드 값을 정수로 반환합니다. 지정한 시간까지 키보드 입력이 없으면 `-1` 을 반환합니다. delay 인자에 `0` 을 전달하면 대기 시간을 무한대로 하겠다는 의미이므로 키를 누를 때까지 프로그램은 멈추고 이때는 `-1` 을 반환할 일은 없습니다.

키보드에서 어떤 키를 눌렀는지를 알아내려면 `cv2.waitKey()` 함수의 반환 값을 출력해 보면 됩니다.

```py
key = cv2.waitKey(0)
print(key)
```

출력되는 키 값을 확인해 보면 ASCII 코드와 같다는 것을 알 수 있습니다. 환경에 따라 한글 모드에서 키를 입력하면 오류가 발생할 수 있으니 키를 입력할 때 한글은 사용하지 않는 것이 좋습니다.

입력된 키를 특정 문자와 비교할 때는 파이썬 기본 함수인 `ord()` 함수를 사용하면 편리합니다. 예를 들어 키보드의 'a' 키를 눌렀는지 확인하기 위한 코드는 다음과 같습니다.

```py
if cv2.waitKey(0) == ord('a'):
```

그런데 몇몇 64비트 환경에서 `cv2.waitKey()` 함수는 8비트(ASCII 코드 크기)보다 큰 32비트 정수를 반환해서 그 값을 `ord()` 함수를 통해 비교하면 서로 다른 값으로 판단할 때가 있습니다. 그래서 하위 8비트를 제외한 비트를 지워야 하는 경우가 있습니다. 0xFF는 하위 8비트가 모두 1로 채워진 숫자이므로 이것과 & 연산을 수행하면 하위 8비트보다 높은 비트는 모두 0으로 채울 수 있습니다.

```py
key = cv2.waitKey(0) & 0xFF
if key == ord('a') :
```

다음 예제는 화면에 이미지를 표시하고 키보드의 'a', 'w', 's', 'd' 키를 누르면 창의위치가 좌, 상, 하, 우 방향으로 10픽셀씩 움직이고, 'esc' 키 또는 'q' 키를 누르면 종료되는 코드입니다.

```py
'''키 이벤트'''
import cv2

img_file = "./img/wonyoung.jpg"
img = cv2.imread(img_file)
title = 'IMG'                   # 창 이름
x, y = 100, 100                 # 최초 좌표

while True:
    cv2.imshow(title, img)
    cv2.moveWindow(title, x, y)
    key = cv2.waitKey(0) & 0xFF # 키보드 입력을 무한 대기, 8비트 마스크처리
    print(key, chr(key))        # 키보드 입력 값, 문자 값 출력
    if key == ord('a'):         # 'a' 키 이면 좌로 이동
        x -= 10
    elif key == ord('s'):       # 's' 키 이면 아래로 이동
        y += 10
    elif key == ord('w'):       # 'w' 키 이면 위로 이동
        y -= 10
    elif key == ord('d'):       # 'd' 키 이면 오른쪽으로 이동
        x += 10
    elif key == ord('q') or key == 27: # 'q' 이거나 'esc' 이면 종료
        break
        cv2.destroyAllWindows()
    cv2.moveWindow(title, x, y )   # 새로운 좌표로 창 이동

[output]
97 a
119 w
120 x
100 d
113 q
```



## 2. 마우스 이벤트

마우스에서 입력을 받으려면 이벤트를 처리할 함수를 미리 선언해 놓고 `cv2.setMouseCallback()` 함수에 그 함수를 전달합니다. 코드로 간단히 묘사하면 다음과 같습니다.

```py
def onMouse(event, x, y, flags, param):
    # 여기에 마우스 이벤트에 맞게 해야 할 작업을 작성합니다.
    pass

cv2.setMouseCallback('title', onMouse)
```

이 두 함수의 모양은 아래와 같습니다.

* `cv2.setMouseCallback(win_name, onMouse [, param])` : onMouse 함수를 등록
    * `win_name` : 이벤트를 등록할 윈도 이름
    * `onMouse` : 이벤트 처리를 위해 미리 선언해 놓은 콜백 함수
    * `param` : 필요에 따라 onMouse 함수에 전달할 인자
* `MouseCallback(event, x, y, flags, param)` : 콜백 함수 선언부
    * `event` : 마우스 이벤트 종류, cv2.EVENT_로 시작하는 상수(12가지)
        * `cv2.EVENT_MOSEMOVE` : 마우스 움직임
        * `cv2.EVENT_LBUTTONDOWN` : 왼쪽 버튼 누름
        * `cv2.EVENT_RBUTTONDOWN` : 오른쪽 버튼 누름
        * `cv2.EVENT_MBUTTONDOWN` : 가운데 버튼 누름
        * `cv2.EVENT_LBUTTONUP` : 왼쪽 버튼 뗌
        * `cv2.EVENT_RBUTTONUP` : 오른쪽 버튼 뗌
        * `cv2.EVENT_MBUTTONUP` : 가운데 버튼 뗌
        * `cv2.EVENT_LBUTTONDBLCLK`: 왼쪽 버튼 더블 클릭
        * `cv2.EVENT_RBUTTONDBLCLK` : 오른쪽 버튼 더블 클릭
        * `cv2.EVENT_MBUTTONDBLCLK` : 가운데 버튼 더블 클릭
        * `cv2.EVENT_MOUSEWHEEL` : 휠 스크롤
        * `cv2.EVENT_MOUSEHWHEEL` : 휠 가로 스크롤
    * `x, y` : 마우스 좌표
    * `flags` : 마우스 동작과 함께 일어난 상태, `cv2.EVENT_FLAG_` 로 시작하는 상수(6가지)
        * `cv2.EVENT_FLAG_LBUTTON(1)` : 왼쪽 버튼 누름
        * `cv2.EVENT_FLAG_RBUTTON(2)` : 오른쪽 버튼 누름
        * `cv2.EVENT_FLAG_MBUTTON(4)` : 가운데 버튼 누름
        * `cv2.EVENT_FLAG_CTRLKEY(8)` : Ctrl 키 누름
        * `cv2.EVENT_FLAG_SHIFTKEY(16)` : Shift 키 누름
        * `cv2.EVENT_FLAG_ALTKEY(32)` : Alt 키 누름
    * `param` : `cv2.setMouseCallback()` 함수에서 전달한 인자

다음은 마우스를 클릭하면 지름이 30픽셀인 동그라미를 그리는 예제입니다.

```py
'''마우스 이벤트로 동그라미 그리기'''
import cv2

title = 'mouse event'                       # 창 제목
img = cv2.imread('./img/blank_500.jpg')     # 백색 이미지 읽기
cv2.imshow(title, img)                      # 백색 이미지 표시

def onMouse(event, x, y, flags, param):     # 마우스 콜백 함수 구현 ---①
    print(event, x, y, )                    # 파라미터 출력
    if event == cv2.EVENT_LBUTTONDOWN:      # 왼쪽 버튼 누름인 경우 ---②
        cv2.circle(img, (x,y), 30, (0,0,0), -1) # 지름 30 크기의 검은색 원을 해당 좌표에 그림
        cv2.imshow(title, img)              # 그려진 이미지를 다시 표시 ---③

cv2.setMouseCallback(title, onMouse)        # 마우스 콜백 함수를 GUI 윈도우에 등록 ---④

while True:
    if cv2.waitKey(0) & 0xFF == 27:         # esc로 종료
        break
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202866618-39ae1bd5-03db-4827-9a2a-8cfb3b558b12.png">
</p>

위 예제의 코드 ①에서 마우스 이벤트를 처리하기 위한 함수를 구현하고 있습니다. 이 함수를 코드 ④에서 윈도에 등록하고 있습니다. 이 함수의 주요 내용은 코드 ②에서처럼 왼쪽 버튼이 눌려지는 것을 알아내는 것입니다. 또 하나 주의해야 할 것은 코드 ③에서처럼 이벤트 내에서 그리기를 했다면 반드시 그림이 그려진 이미지를 다시 화면에 표시해야 한다는 것입니다.

아래의 코드는 이벤트 처리 함수의 선언부인데, 모두 5개의 인자를 선언해야 합니다.

```py
def onMouse(event, x, y, flags, param):
```

함수 내부에서 사용하지 않더라도 5개의 인자는 모두 선언부에 기재해야 하며, 그렇지 않으면 오류가 발생합니다. 이 함수의 첫 번째 인자 `event` 는 발생한 이벤트의 종류를 나타내는 것으로 `cv2.EVENT_` 로 시작하는 상수 값 중에 하나입니다. API에 선언되어 있는 모든 이벤트 상수에 대응하는 이벤트는 코드를 실행하는 환경에 따라 작동하지 않는 경우도 있으니 주요한 것 위주로 사용하는 것이 좋습니다. `x, y` 는 이벤트가 발생한 마우스의 좌표입니다. `flags` 는 이벤트가 발생할 때 키보드나 마우스의 추가적인 상태를 알려줍니다. 이 값과 비교할 상수는 이름이 `cv2.EVENT_FALG_` 로 시작하는 선언되어 있는 상수들입니다.

이 플래그는 시프트 키와 컨트롤 키를 함께 누른 상태처럼 여러 가지 상태를 하나의 값으로 한꺼번에 나타낼 수 있어야 합니다. 그래서 선언된 상수들이 실제로 갖는 값은 0, 1, 2, 3, 4처럼 순차적으로 증가하는 값이 아니라 1, 2, 4, 8, 16, 32 순으로 2진수 비트 자릿수에 맞는 값을 각각 갖습니다. 따라서 함수의 인자로 전달되는 값은 여러 상태를 나타내는 값을 조합한 값으로, 어떤 상태인지 알기 위해서는 비트 단위 '&'(논리 곱) 또는 '|'(논리합) 연산을 써서 알아내야 합니다.

예를 들어 flag 값이 8이라면 `cv2.EVENT_FLAG_CTRLKEY` 의 값과 같습니다. 이런 경우 flag 값과 관심 있는 상수를 비교해서 맞으면 컨트롤 키가 눌러진 상태로 판단하면 됩니다.

하지만, 만약 flag 값이 25라면 어떤 플래그 상수와 비교해도 맞는 것을 찾을 수 없습니다. 이 경우 25 = 1 + 8 + 16이므로 1, 8, 16에 맞는 플래그 상수와 따로따로 비교해서 찾아내야 합니다. 이것은 각각 `cv2.EVENT_FLAG_LBUTTON`, `cv2.EVENT_FLAG_CTRLKEY`, `CV2.EVENT_FLAG_SHIFTKEY` 에 해당합니다.

flags로부터 상태를 각각 알아내는 방법은 다음 코드와 같습니다.

```py
if flags & cv2.EVENT_FLAG_LBUTTON:
    pass    # 마우스 왼쪽 버튼 눌림
if flags & cv2.EVENT_FLAG_CTRLKEY:
    pass    # 컨트롤 키눌림
if flags & cv2.EVENT_FLAG_SHIFTKEY:
    pass    # 시프트 키눌림
```

결국 관심 있는 상태 플래그 값과 인자값을 & 연산하면 됩니다. 그러면 각각의 조건문에 모두 True로 반환되어 처리됩니다.

[예제 2-18]은 앞서 다룬 마우스로 동그라미 그리기 예제를 컨트롤 키를 누르면 빨간색으로, 시프트 키를 누르면 파란색으로, 시프트 키와 컨트롤 키를 동시에 누르면 초록색으로 그리게 수정한 것입니다.


```py
'''플래그를 이용한 동그라미 그리기'''
import cv2

title = 'mouse event'                       # 창 제목
img = cv2.imread('./img/blank_500.jpg')    # 백색 이미지 읽기
cv2.imshow(title, img)                      # 백색 이미지 표시

colors = {'black' : (0,0,0),
          'red' : (0,0,255),
          'blue' : (255,0,0),
          'green' : (0,255,0)}   # 색상 미리 정의

def onMouse(event, x, y, flags, param): # 아무스 콜백 함수 구현 ---①
    print(event, x, y, flags)           # 파라미터 출력
    color = colors['black']
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누름인 경우 ---②
        # 컨트롤키와 쉬프트 키를 모두 누른 경우
        if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY : 
            color = colors['green']
        elif flags & cv2.EVENT_FLAG_SHIFTKEY :  # 쉬프트 키를 누른 경우
            color = colors['blue']
        elif flags & cv2.EVENT_FLAG_CTRLKEY :   # 컨트롤 키를 누른 경우
            color = colors['red']
        # 지름 30 크기의 검은색 원을 해당 좌표에 그림
        cv2.circle(img, (x,y), 30, color, -1) 
        cv2.imshow(title, img)          # 그려진 이미지를 다시 표시 ---③

cv2.setMouseCallback(title, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록 ---④

while True:
    if cv2.waitKey(0) & 0xFF == 27:     # esc로 종료
        break
cv2.destroyAllWindows()
```

<br>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/202867028-d8f2bb6f-141d-4e0f-9e78-fa66c66ab1bc.png">
</p>




