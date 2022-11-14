---
layout: post
title: OpenCV 개요와 설치
category: OpenCV
tag: OpenCV
---

[![Hits](https://hits.sh/museonghwang.github.io.svg?view=today-total&style=for-the-badge&label=Visitors&color=007ec6)](https://hits.sh/museonghwang.github.io/)

<br>

해당 게시물은 [파이썬으로 만드는 OpenCV 프로젝트(이세우 저)](https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/README.md) 를 바탕으로 작성되었습니다.





# 영상 처리와 컴퓨터 비전



## 영상 처리

**영상 처리(image processing)**는 카메라로 찍은 사진 또는 영상에 여러 가지 연산을 가해서 원하는 결과를 새롭게 얻어내는 과정입니다. 대부분 영상 처리의 목적은 더 좋은 품질의 영상을 얻으려는 것입니다. 몇 가지 예를 들면 다음과 같습니다.
* 영상(화질) 개선: 사진이나 동영상이 너무 어둡거나 밝아서 화질을 개선하는 과정
* 영상 복원: 오래되어 빛바랜 옛날 사진이나 영상을 현대적인 품질로 복원하는 과정
* 영상 분할: 사진이나 영상에서 원하는 부분만 오려내는 과정



## 컴퓨터 비전

**컴퓨터 비전**은 영상 처리 개념을 포함하는 좀 더 큰 포괄적인 의미입니다. 영상 처리가 원본 영상을 사용자가 원하는 새로운 영상으로 바꿔 주는 기술이라면 컴퓨터 비전은 영상에서 의미 있는 정보를 추출해 주는 기술을 말합니다. 예를 들면 다음과 같습니다.
* 객체 검출(object detection): 영상 속에 원하는 대상이 어디에 있는지 검출
* 객체 추적(object tracking): 영상 속 관심 있는 피사체가 어디로 움직이는지 추적
* 객체 인식(object recognition): 영상 속 피사체가 무엇인지 인식.

일반적으로 컴퓨터 비전 작업은 입력받은 원본 영상을 영상 처리하여 원하는 품질의 결과 영상을 얻어낸 다음, 컴퓨터 비전으로 원하는 정보를 얻어내는 과정이 반복적으로 일어납니다.

<br>





# OpenCV



## OpenCV 개요

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/201581818-ada0f452-417e-42ea-9226-20a77b420c39.png">
</p>

OpenCV는 오픈 소스 컴퓨터 비전 라이브러리(Open Source Computer Vision Library)를 줄여 쓴 말로, OpenCV는 영상 처리와 컴퓨터 비전 프로그래밍 분야의 가장 대표적인 라이브러리입니다.

OpenCV는 사진 혹은 영상을 처리해주는 포토샵 기능을 프로그래밍 언어로 구현할 수 있게 해주는 라이브러리라고 생각해도 크게 틀리지 않습니다.

OpenCV의 공식 웹사이트와 문서는 다음과 같습니다.
* https://opencv.org/
* https://docs.opencv.org/4.x/index.html

OpenCV의 소스 코드 저장소는 다음과 같이 2개로 나뉩니다.
* 메인 저장소 : https://github.com/opencv/opencv
* 엑스트라(extra) 저장소 : https://github.com/opencv/opencv_contrib

메인 저장소에서는 OpenCV 공식 배포에 사용하는 코드를 관리합니다. 엑스트라 저장소는 컨트리브(contrib) 저장소라고도 하는데, 아직 알고리즘이나 구현의 성숙도가 떨어지거나 대중화되지 않은 내용을 포함하고 있으며, 향후 완성도가 높아지면 메인 저장소로 옮겨집니다.



## M1 Mac에서 OpenCV 설치

M1 Mac에서 OpenCV 설치를 진행해보겠습니다. OpenCV 학습을 위하여 다음과 같이 개발환경을 세팅했습니다.

```
Python 3.6
Numpy 1.14
OpenCV-Python 3.4.1, 엑스트라(contrib) 포함
Matplotlib 2.2.2
```



### 1. Anaconda 설치

효율적으로 가상환경을 관리하기위해 anaconda를 먼저 설치하겠습니다. 다음과 같이 anaconda 홈페이지에 들어가서 자신의 운영체제에 맞는 Installer를 다운받아 설치하는 방법이 있습니다.
* https://www.anaconda.com/products/distribution

하지만 여기에서는 homebrew로 conda를 설치하겠습니다.
```py
brew install anaconda
```

터미널에서 아래의 명령어를 입력하여 제대로 설치되었는지 확인합니다.
```py
conda
```

만약 `zsh: command not found: conda` 라는 에러가 발생할 경우, 설치경로를 찾아 conda init zsh를 입력 후 쉘을 재시작하면 됩니다.
```py
/opt/homebrew/anaconda3/bin/conda init zsh
```

이후 다음과 같이 나오면 anaconda 설치완료입니다.
```py
> conda -V
conda 22.9.0
```

또한 conda 설치시 기본 파이썬 버전이 3.9로 바뀌고, conda base 환경을 쓰게 됩니다. 이 점 주의해주세요.
```py
> python --version
Python 3.9.13
```



### 2. 가상환경 만들기

우선, 파이썬 3.6 버전을 갖는 가상 환경을 만들어줍니다.
```py
# Anaconda 환경에서 가상환경 만들기
conda create -n opencv python=3.6
```

파이썬 3.6 버전이 설치된 opencv라는 이름의 가상 환경이 만들어졌습니다. 이렇게 만든 가상 환경을 아래와 같이 실행해 줍니다.
```py
# 가상환경 실행
conda activate opencv
```

가상 환경 안에 원하는 버전의 모듈을 설치합니다.
```py
# numpy 1.14 버전 설치
pip3 install numpy==1.14.0

# 엑스트라(contrib)를 포함한 OpenCV-Python 모듈 3.4.1 설치
pip3 install opencv-contrib-python==3.4.1.15

# matplotlib 2.2.2 버전 설치
pip3 install matplotlib==2.2.2
```

이제, prompt 창에서 python을 실행시킨 뒤 아래와 같이 입력하면 각 모듈별 버전이 뜰 겁니다. 다른 버전이 뜨면 설치가 잘못되었거나 가상 환경 세팅이 잘못된 겁니다.
```py
>>> import numpy
>>> numpy.__version__
'1.14.0'


>>> import cv2
>>> cv2.__version__
'3.4.1'

>>> import matplotlib
>>> matplotlib.__version__
'2.2.2'
```

이상으로 OpenCV의 개요와 설치 방법에 대해 알아봤습니다.




