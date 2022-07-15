---
title: ImageNet Classification with Deep Convolutional Neural Networks
category: Classification Paper
tag: classification-paper
date: 2022-07-15
---     

<p align="center">
<img width="1865" alt="image" src="https://user-images.githubusercontent.com/77891754/179117949-a48424ab-6f2a-4a8a-a26e-3f15426e09bd.png">
</p>


# Abstract

ImageNet LSVRC-2010 대회에서 120만 개의 high-resolution images를 1000개의 다른 클래스로 분류하기 위해 크고 깊은 컨볼루션 신경망을 훈련했습니다. 테스트 데이터에서 top-1 error rates 37.5% 및 top-5 error rates 17.0%를 달성하여 이전의 SOTA보다 상당히 개선되었습니다.

6천만개의 parameters와 65만개의 neurons이 있는 신경망은 5개의 컨볼루션 계층으로 구성되며, 그 중 일부는 max-pooling layers가 뒤따르고, 최종 1000-way softmax가 있는 3개의 fully-connected layers가 뒤따릅니다. 훈련을 더 빠르게 하기 위해 non-saturating neurons과 합성곱 연산에 매우 효율적인 GPU 구현을 사용했고, 특히 fully-connected layers에서 overfitting을 줄이기 위해 "Dropout"이라는 최근에 개발된 regularization 방법을 사용했습니다.

또한 ILSVRC-2012 대회에서 이 모델의 변형을 입력했고, 2위를 차지한 26.2%와 비교하여 top-5 test error rate 15.3%로 우승했습니다.





# Introduction

객체 인식(object recognition)에 대한 현재 접근 방식은 기계 학습 방법을 필수적으로 사용합니다. 성능을 개선하기 위해 더 큰 datasets를 수집하고, 더 강력한 models을 학습하며, overfitting을 방지하기 위한 더 나은 기술을 사용할 수 있습니다. 최근까지의 datasets는 이미지 수만 개 정도로 비교적 작았으며, Simple recognition tasks는 이정도 크기의 datasets와 augmented with label-preserving transformations로 꽤 잘 해결될 수 있습니다. 예를 들어, MNIST digit-recognition task(<0.3%)에서 현재 error rate은 인간의 성능에 도달했습니다. 그러나 실제환경에서의 objects들은 상당한 변동성을 보이기 때문에, 객체를 인식하는 방법을 배우려면 훨씬 더 많은 training set를 사용하는것이 필수적입니다. 그리고 실제로 small image datasets의 단점은 지금까지 널리 인식되었지만, 최근에 들어서야 수백만 장의 라벨된 이미지를 포함하는 데이터 세트를 수집하는 것이 가능해졌습니다. 더 큰 새로운 데이터 세트는 수십만 개의 완전히 분할된 이미지로 구성된 LabelMe와 22,000개 이상의 범주에서 1,500만 개 이상의 레이블이 지정된 고해상도 이미지로 구성된 ImageNet를 포함합니다.

수백만 개의 이미지에서 수천 개의 objects에 대해 모델이 학습하기 위해서는 모델의 학습 수용력(learning capacity)이 높아야합니다. 그러나 object recognition task의 엄청난 복잡성 때문에 ImageNet과 같은 큰 데이터 세트를 사용하더라도 해결하기 어려우므로, 우리가 가지고 있지 않은 데이터까지 처리하기 위해서는 우리의 모델은 많은 사전 지식을 알고 있어야 합니다. CNN(Convolutional Neural Networks)은 학습 수용력(learning capacity)이 높은 종류의 모델 중 하나로, 수용력(capacity)은 깊이와 너비를 변경하여 제어할 수 있으며, 이미지의 특성(namely, stationarity of statistics and locality of pixel dependencies)에 대해 강력하고 대부분 정확한 추정을 합니다.

따라서 비슷한 크기의 레이어가 있는 standard feedforward neural networks과 비교할 때 CNN은 훨씬 적은 connections과 parameters를 가지고 있기 때문에 훈련하기가 더 쉽고 빠르지만, 이론적으로 가장 좋은 성능은 약간만 떨어질 수 있습니다.

CNN의 매력적인 특성과 local architecture의 상대적 효율성에도 불구하고, high-resolution images에 대규모로 적용하는 데는 여전히 엄청난 비용이 듭니다. 운 좋게도 고도로 최적화된 2D 컨볼루션 구현과 결합된 현재 GPU는 흥미롭게도 큰 CNN의 교육을 용이하게 할 만큼 강력하며 ImageNet과 같은 최근 데이터 세트에는 심각한 과적합 없이 이러한 모델을 훈련할 수 있는 충분한 레이블이 지정된 예제가 포함되어 있습니다. 다행히도, 2D 합성곱을 효율적으로 처리하는 최근의 GPU는 큰 규모의 합성곱 신경망을 학습시킬만큼 강력하며, 또한 ImageNet과 같은 대형 데이터셋은 오버피팅을 방지할 정도로 많은 수의 라벨된 이미지를 제공합니다.


이 논문의 기여는 다음과 같습니다.

- ILSVRC-2010 및 ILSVRC-2012 대회에서 사용된 ImageNet의 subset에 대해 현재까지 가장 큰 CNN 중 하나를 훈련했으며, SOTA의 결과를 달성했습니다.
- 2D 합성곱에 고도로 최적화된 GPU 구현과, CNN 훈련에 내재된 다른 모든 작업을 작성했으며 이를 공개합니다.
- 네트워크에는 성능을 향상시키고 학습시간을 줄이는 여러 가지의 새롭고 특이한 특징(feature)들을 포함하는데, Section 3에 자세히 설명되어 있습니다.
- 우리의 네트워크 size는 overfitting을 심각한 문제로 만들었으므로 Section 4에서 overfitting을 방지하기 위해 몇 가지 효과적인 기술을 사용했습니다.
- 최종 네트워크에는 5개의 convolutional layers과 3개의 fully-connected layers가 포함되어 있으며, 이 depth가 중요한 것 같습니다: 컨볼루션 레이어(각각은 모델 매개변수의 1% 이하를 포함)를 제거하면 성능이 저하된다는 것을 발견했습니다.
- 결국 network의 size는 주로 현재 GPU에서 사용 가능한 메모리 양과 우리가 기꺼이 허용할 수 있는 훈련 시간에 의해 제한됩니다. 본 네트워크는 2개의 GTX 580 3GB GPU에서 훈련하는 데 5~6일이 걸립니다.





# The Dataset

ImageNet은 대략 22000개의 카테고리에 속하는 1500만 개 이상의 라벨된 고해상도 이미지의 데이터세트입니다. 이미지는 웹에서 수집되었으며 Amazon의 Mechanical Turk 크라우드 소싱 도구를 사용하여 사람이 라벨을 붙였습니다. 2010년부터 Pascal Visual Object Challenge의 일환으로 ImageNet Large-Scale Visual Recognition Challenge(ILSVRC)대회가 매년 개최되었습니다. ILSVRC는 1000개 범주 각각에 약 1000개 이미지가 있는 ImageNet의 subset을 사용합니다. 전체적으로 약 120만 개의 훈련 이미지, 50,000개의 검증 이미지, 150,000개의 테스트 이미지가 있습니다.

ILSVRC-2010은 테스트 세트 라벨을 사용할 수 있는 유일한 ILSVRC 버전이므로, 해당 버전으로 대부분의 실험을 수행했다. 또한 우리는 ILSVRC-2012 대회에도 모델에 참가했으므로 섹션 6에서 테스트 세트 레이블을 사용할 수 없는 이 버전의 데이터 세트에 대한 결과도 보고합니다.

ImageNet에서는 두 가지 에러율을 기록하는데, 이미지를 입력했을 때 모델이 정답으로 예측한 라벨이 오답인 비율(top-1 error rate)과 가능성이 가장 높다고 예측한 5가지 라벨이 모두 오답인 비율(top-5 error rate)가 사용됩니다. ImageNet 데이터셋은 다양한 해상도의 이미지를 포함하는데, 우리의 모델은 고정된 크기의 이미지를 입력해야 한다. 따라서 이미지를 256 × 256의 고정 해상도(resolution)로 다운 샘플링(down-sampled)했습니다. 직사각형 이미지가 주어지면 먼저 짧은 변의 길이를 256로 맞춘 다음, 이미지의 가운데 부분을 256x256 크기로 패치를 잘라냈습니다. 그리고 각 픽셀에서 training set에 대한 mean activity을 빼는 것을 외에는 다른 방식으로 이미지 전처리를 하지 않았습니다. 그래서 우리는 픽셀의 raw RGB값에 대해 네트워크를 훈련했습니다.





# The Architecture

우리의 신경망은 레이어 8개(합성곱층 5개와 전연결층 3개)로 이루어져있다. 우리 신경망 구조의 새로운 특징들을 아래에 서술하였다. 중요한 순서대로 위에서부터 배치하였다.

네트워크의 아키텍처는 8개의 학습된 layers가 포함되어 있는데, 5개는 convolutional layer이고 3개는 fully-connected layer입니다. 아래에는 네트워크 아키텍처에서 참신하거나 특이한 기능 중 일부를 설명합니다. 중요도에 따라 가장 중요한 것부터 배치했습니다.



## ReLU Nonlinearity

뉴런의 output $f$를 input $x$의 함수(즉, 활성화함수)로 모델링하는 일반적인 방법은 $f(x) = tanh(x)$ 또는 $f(x) = (1 + e^(−x))^(−1)$입니다. 경사 하강법을 사용한 훈련 시간 측면에서, 이러한 포화하는 비선형 함수(saturating linearities)는 비포화 비선형 함수(non-saturating linearity)인 ReLU($f(x) = max(0, x)$)보다 훨씬 느립니다. Nair와 Hinton에 따르면 ReLUs(Rectified Linear Units)라고 불리는 이 비선형성(nonlinearity)을 가진 뉴런을 참조합니다. ReLU를 사용하는 Deep convolutional neural networks은 tanh 함수를 사용했을 때보다 몇 배 더 빠르게 훈련합니다.

Figure 1에서, 특정 four-layer convolutional network에 대한 CIFAR-10 데이터 세트에서 25% training error에 도달하는 데 필요한 반복 횟수를 보여줍니다. 이 결과를 봤을 때, 전통적인 포화 뉴런 모델을 사용했다면 이 작업을 위한 큰 규모의 신경망을 지금까지 실험할 수 없었을지도 모른다라는 것을 보여준다.


<p align="center">
<img width="928" alt="image" src="https://user-images.githubusercontent.com/77891754/179141726-21960cf6-4f2a-4e1c-9ae5-7d8509bbbd93.png" style="zoom:60%;">
</p>

<p align="center" style="font-size:80%">
4-layer convolutional neural network에서 ReLU(실선)가 tanh 뉴런(점선)보다 CIFAR-10에서 6배 더 빠르게 25% training error rate에 도달합니다. 각 네트워크에 대한 learning rates은 가능한 한 빨리 훈련할 수 있도록 독립적으로 선택되었습니다. 어떤 종류의 regularization도 사용되지 않았습니다. 여기에서 설명하는 효과의 크기는 네트워크 아키텍처에 따라 다르지만 ReLU가 있는 네트워크가 saturating 뉴런이 있는 네트워크보다 일관되게 몇 배 더 빠르게 학습합니다.
</p>


물론 이런 새로운 활성함수를 고려한게 AlexNet이 처음은 아닙니다. 기존에 Jarrett은 Caltech-101 데이터셋에 대해 local average pooling으로 정규화 했을 경우 f(x) = |tanh(x)| 가 부분적으로 효과적이였다고 주장했습니다. 하지만 여기서는 overfitting을 막는 데에 초점이 맞춰져 있었기 때문에 우리가 발표한 ReLU를 사용할 때 fit 하기위해 발생되는 accelerated ability와는 다릅니다. 즉 AlexNet에서는 오버피팅 방지가 아니라 빠른 학습을 요구하기 때문에 ReLU를 사용하였습니다.

**빠르게 학습하는것은 대규모 데이터 세트에서 훈련된 대규모 모델의 성능에 큰 영향을 미칩니다.**


 
## Training on Multiple GPUs

GPU 하나의 메모리가 3GB로 너무 작기 때문에 두 개를 병행하여 사용하였다. 당시 GPU도 서로 다른 GPU의 메모리에 직접 읽고 쓰기가 가능했기 때문에 cross-GPU parallelization에 용이했다. 여기서 적용한 병렬화 전략은 커널(혹은 뉴런)의 절반만큼 각각의 GPU가 담당하게 하는 것이다. 그리고 특정 layer에서만 GPU가 서로 상호작용한다. 만약 이 특정 layer가 layer2이라면 이를 input으로 받는 layer3에서의 뉴런들은 layer2의 모든 kernel map(feature map일 것으로 추정)들로부터 입력을 받는다. 그러나 layer4의 뉴런들은 layer3에서 자신의 GPU에 할당된 부분에서만 kernel map을 입력으로 받는다. 이 상호작용의 패턴을 알아내는 것은 문제가 될 수 있지만, 이를 통해 전체 상호작용의 횟수(?)를 설정할 수 있고, 더 나아가 원하는 연산을 할 때까지 조절할 수 있다.

두 개의 GPU를 사용하는 기법으 error rate를 top-1에서는 1.7%, top-5에서는 1.2가량 줄였으며, 속도도 하나만 사용하는 것보다 조금 더 빨랐다.

-----


단일 GTX 580 GPU에는 3GB의 메모리만 있으므로 훈련할 수 있는 네트워크의 최대 크기가 제한됩니다. 1.2백만개의 training 데이터를 사용하기에 1개의 GPU로는 충분치 않아서 두 개의 GPU를 병행하여 사용하였습니다.

현재 GPU는 서로 다른 GPU의 메모리에 직접 읽고 쓰기가 가능하기 때문에 cross-GPU parallelization에 적합합니다. 여기서 적용한 병렬화 전략은 커널(혹은 뉴런)의 절반만큼 각각의 GPU가 담당하게 하고, 특정 layer에서만 GPU가 서로 상호작용하게끔 합니다. 예를 들어 layer 3의 kernel은 layer 2의 모든 kernel map에서 입력을 받습니다. 그러나 layer 4의 kernel은 동일한 GPU에 있는 layer 3의 kernel map에서만 입력을 받습니다. 만약 이 특정 layer가 layer2이라면 이를 input으로 받는 layer3에서의 뉴런들은 layer2의 모든 kernel map(feature map일 것으로 추정)들로부터 입력을 받는다. 그러나 layer4의 뉴런들은 layer3에서 자신의 GPU에 할당된 부분에서만 kernel map을 입력으로 받습니다. 연결패턴을 선택하는 것은 교차 검증의 문제이지만, 이를 통해 통신량이 연산량의 허용 가능한 부분이 될 때까지 통신량을 정확하게 조정할 수 있습니다.

결과 아키텍처는 열이 독립적이지 않다는 점을 제외하고 "기둥형" CNN의 아키텍처와 다소 유사합니다. 두 개의 GPU를 사용할때 error rate를 top-1에서는 1.7%, top-5에서는 1.2가량 줄였으며, 속도도 하나만 사용하는 것보다 조금 더 빨랐습니다.



## Local Response Normalization




## Overlapping Pooling



## Overall Architecture








# Reducing Overfitting
## Data Augmentation
## Dropout







# Details of learning






# Results





# Discussion





