---
title: ImageNet Classification with Deep Convolutional Neural Networks
category: Classification Paper
tag: classification-paper
date: 2022-07-15
---     


<p align="center">
<img width="1773" alt="image" src="https://user-images.githubusercontent.com/77891754/179393371-b4703b24-49e7-48ea-bbed-9e6ed03ae118.png">
</p>



<p align="center">
<img width="1870" alt="image" src="https://user-images.githubusercontent.com/77891754/179395389-5c8e8899-deb6-4a61-9a5b-d0ebe0d0f64b.png">
</p>

# Abstract

ImageNet LSVRC-2010 대회에서 120만 개의 high-resolution images를 1000개의 다른 클래스로 분류하기 위해 크고 깊은 convolutional network을 훈련했으며, test data에서 top-1 error rates 37.5% 및 top-5 error rates 17.0%를 달성하여 이전의 SOTA보다 상당히 개선되었습니다.

6천만개의 parameters와 65만개의 neurons이 있는 신경망은 5개의 convolutional layers로 구성되며, 그 중 일부는 max-pooling layers가 뒤따르고, 최종 1000-way softmax가 있는 3개의 fully-connected layers가 뒤따릅니다. 훈련을 더 빠르게 하기 위해 non-saturating neurons과 합성곱 연산에 매우 효율적인 GPU 구현을 사용했고, 특히 fully-connected layers에서 overfitting을 줄이기 위해 "Dropout"이라는 최근에 개발된 regularization 방법을 사용했습니다. 또한 ILSVRC-2012 대회에서 이 모델의 변형을 입력했고, 2위를 차지한 26.2%와 비교하여 top-5 test error rate 15.3%로 우승했습니다.





# Introduction

객체 인식(object recognition)에 대한 현재 접근 방식은 기계 학습 방법을 필수적으로 사용하는데, 성능을 개선하기 위해 더 큰 datasets를 수집하고, 더 강력한 models을 학습하며, overfitting을 방지하기 위한 더 나은 기술을 사용할 수 있습니다. 최근까지의 라벨이 된 datasets는 이미지 수만 개 정도로 비교적 작았는데, Simple recognition tasks는 이정도 size의 datasets으로 꽤 잘 해결될 수 있었습니다. 그러나 실제환경에서의 objects들은 상당한 변동성을 보이기 때문에, 객체를 인식하는 방법을 배우려면 훨씬 더 많은 training set를 사용하는것이 필수적입니다. 그리고 실제로 small image datasets의 단점은 지금까지 널리 인지되어왔지만, 최근에 들어서야 수백만 장의 라벨이 된 이미지를 포함하는 데이터 세트를 수집하는 것이 가능해졌습니다.


또한 수백만 개의 이미지에서 수천 개의 objects에 대해 모델이 학습하기 위해서는 모델의 학습 수용력(learning capacity)이 높아야합니다. CNN(Convolutional Neural Networks)은 학습 수용력(learning capacity)이 큰 종류의 모델 중 하나입니다. CNN의 수용력(capacity)은 깊이와 너비를 달리하여 제어할 수 있으며, 이미지의 nature(특성, 성질)(즉, stationarity of statistics and locality of pixel dependencies)에 대해 강력하고 대부분 정확한 추정을 합니다. 따라서 비슷한 크기의 layer가 있는 standard feedforward neural networks과 비교할 때 CNN은 훨씬 적은 connections과 parameters를 가지고 있기 때문에 훈련하기가 더 쉽고 빠르지만, 이론적으로 가장 좋은 성능은 약간 떨어질 수 있습니다.


CNN의 매력적인 특성과 local architecture의 상대적 효율성에도 불구하고, high-resolution images에 대규모로 적용하는 데는 여전히 엄청난 비용이 듭니다. 다행히도, 2D 합성곱을 효율적으로 처리하는 최근의 GPU는 큰 규모의 합성곱 신경망을 학습시킬만큼 강력하며, 또한 ImageNet과 같은 대형 데이터셋은 overfitting을 방지할 정도로 많은 수의 라벨이 된 이미지를 제공합니다.


이 논문의 기여는 다음과 같습니다.

- ILSVRC-2010 및 ILSVRC-2012 대회에서 사용된 ImageNet의 subset에 대해 현재까지 가장 큰 CNN 중 하나를 훈련했으며, SOTA의 결과를 달성
- 2D 합성곱과 CNN 훈련에 내재된 다른 모든 작업에 고도로 최적화된 GPU 구현을 하였으며 이를 공개
- 본 network는 성능을 향상시키고 학습시간을 줄이는 여러 가지의 새롭고 특이한 특징(feature)들을 포함
- 본 network의 크기는 120만 개의 라벨이 된 training dataset을 가지고도 overfitting문제를 일으켰기 때문에, overfitting을 방지하기 위한 몇 가지 효과적인 기술을 사용
- 최종 network에는 5개의 convolutional layers과 3개의 fully-connected layers가 포함되어 있으며, 이 depth가 중요함
- convolutional layers(각각은 모델 매개변수의 1% 이하를 포함)를 제거하면 성능이 저하된다는 것을 발견
- network의 크기는 주로 현재 GPU에서 사용할 수 있는 메모리 양과 우리가 기꺼이 허용할 수 있는 training time에 의해 제한됨
- 2개의 GTX 580 3GB GPU에 대해 training하는 데 5~6일 소요





# The Dataset

- ImageNet은 대략 22000개의 카테고리에 속하는 1500만 개 이상의 라벨링이 된 고해상도 이미지의 dataset.
- 약 120만 개의 training image, 50,000개의 validation image, 150,000개의 test image 사용.
- ImageNet에서는 2가지의 error rates(top-1 및 top-5)을 사용하는 것이 일반적.
- Test set은 ILSVRC-2010 버전을 주로 사용
- 이미지를 256 × 256의 고정 해상도(resolution)로 다운 샘플링(down-sampled)적용.
- 직사각형 이미지는 짧은 변의 길이가 256이 되도록 rescale 후, 가운데 부분을 256 x 256 크기로 patch를 잘라냄.
- 각 픽셀에서 training set에 대한 mean activity을 빼는 것을 외에는, 다른 방식으로 이미지 전처리를 하지 않았고, 픽셀의 raw RGB값에 대해 네트워크를 train을 진행
- 본 논문에서는 네트워크 입력이 224×224라고 언급했지만 이는 실수이며 입력은 227×227입니다.


<p align="center">
<img width="1132" alt="image" src="https://user-images.githubusercontent.com/77891754/179395691-d42d3754-ae07-49da-8ef3-2cf9926e2257.png" style="zoom:40%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</a></p>





# The Architecture



## ReLU Nonlinearity


<p align="center">
<img width="928" alt="image" src="https://user-images.githubusercontent.com/77891754/179141726-21960cf6-4f2a-4e1c-9ae5-7d8509bbbd93.png" style="zoom:60%;">
</p>

<p align="center" style="font-size:80%">
4-layer convolutional neural network에서 ReLU(실선)가 tanh 뉴런(점선)보다 CIFAR-10에서 6배 더 빠르게 25% training error rate에 도달합니다. 각 네트워크에 대한 learning rates은 가능한 한 빨리 훈련할 수 있도록 독립적으로 선택되었습니다. 어떤 종류의 regularization도 사용되지 않았습니다. 여기에서 설명하는 효과의 크기는 네트워크 아키텍처에 따라 다르지만 ReLU가 있는 네트워크가 saturating 뉴런이 있는 네트워크보다 일관되게 몇 배 더 빠르게 학습합니다.
</p>


AlexNet의 중요한 기능은 ReLU(Rectified Linear Unit) 비선형성을 사용한다는 것입니다. Tanh 또는 Sigmoid 활성화 함수는 신경망 모델을 훈련하는 일반적인 방법이었습니다. AlexNet은 ReLU 비포화(non-saturating) 활성화함수를 사용하면 tanh 또는 sigmoid와 같은 포화(saturating) 활성화 함수를 사용하는 것보다 깊은 CNN을 훨씬 빠르게 훈련할 수 있음을 보여주었습니다. 논문의 그림은 ReLUs(실선 곡선)를 사용하여 AlexNet이 tanh(점선 곡선)를 사용하는 동등한 네트워크보다 6배 빠른 25% 훈련 오류율을 달성할 수 있음을 보여줍니다. 이것은 CIFAR-10 test set에서 테스트되었습니다. 


물론 이런 새로운 활성함수를 고려한게 AlexNet이 처음은 아니지만, AlexNet에서는 오버피팅 방지가 아니라 빠른 학습을 요구하기 때문에 ReLU를 사용하였습니다. 핵심은 더 빠른 학습은 대규모 데이터 세트에 대해 훈련된 대규모 모델의 성능에 큰 영향을 미친다는 것 입니다.


ReLU를 사용하여 더 빨리 훈련하는 이유를 살펴보겠습니다. Tanh와 ReLU 함수는 다음과 같이 주어집니다.


<p align="center">
<img width="1889" alt="image" src="https://user-images.githubusercontent.com/77891754/179395879-b7df0e52-8951-4335-945d-3ae2512f2846.png" style="zoom:80%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</p>


tanh 함수는 z가 조금만 커지거나 작아지면 매우 높거나 매우 낮은 값에서 포화(saturating)됩니다. 이 영역에서 함수의 기울기는 0에 매우 가까운데, 이것은 경사하강법을 늦출 수 있습니다. 반면에 ReLU 함수에서 z가 커져도 기울기가 0으로 가지 않는, 포화상태가 되지 않습니다(non-saturating). 이렇게 하면 최적화가 더 빨리 수렴되는 데 도움이 됩니다. z의 음수 값의 경우 기울기는 여전히 0이지만 신경망의 대부분의 뉴런은 일반적으로 양수 값을 갖게 됩니다. 같은 이유로 ReLU도 시그모이드 함수보다 우위에 있습니다.




## Training on Multiple GPUs


<p align="center">
<img width="1902" alt="image" src="https://user-images.githubusercontent.com/77891754/179396192-ad89867d-e8f7-462d-b853-fa8bc0eff532.png" style="zoom:40%;">
</p>


- Intra GPU Connection : 1,2,4,5 번째 Conv layer에서는 같은 GPU 내에서의 kernel만 사용할 수 있음
- Inter GPU Connection : 3번째 Conv layer와 3개의 FC layer에서는 모든 kernel을 사용할 수 있음
- 두 개의 GPU를 사용했을때 top-1 및 top-5 error rate를 각각 1.7%, 1.2%가량 줄였으며, 속도도 GPU 하나만 사용하는 것보다 조금 더 빨랐습니다.


단일 GTX 580 GPU에는 3GB의 메모리만 있으므로 훈련할 수 있는 네트워크의 최대 크기가 제한됩니다. 1.2백만개의 training 데이터를 네트워크에 훈련시키기에 너무 커서 1개의 GPU로는 충분치 않다라고 판단했고, 두 개의 GPU에 network를 분산시켰습니다. 병렬화 기법은 기본적으로 kernel(또는 뉴런)의 절반을 각각의 GPU에 배치하는데, 여기서 추가적인 trick은 GPU는 특정 layer에서만 connection 하게끔 합니다. 그에 따른 architecture는 열이 독립적이지 않다는 점을 제외하고, Cires가 작성한 "기둥형" CNN과 다소 유사합니다.

<p align="center">
<img width="879" alt="image" src="https://user-images.githubusercontent.com/77891754/179396665-6bfeb276-f67e-45d4-a853-3c9c8e61c047.png" style="zoom:60%;">
</p>

<p align="center" style="font-size:80%">High-Performance Neural Networks for Visual Object Classification(Cires an et al.)</p>




## Local Response Normalization

ReLU는 saturating되지 않도록 input normalization가 필요하지 않은 특성을 가지고 있습니다. 하지만 ReLU의 출력은 입력에 비례하여 그대로 증가가 됩니다. 그렇게 되면 convolution 또는 pooling시 매우 높은 하나의 픽셀값이 주변의 픽셀에 영향을 미치게 됩니다. 이런 상황을 방지하기 위해 다른 activation map의 같은 위치에 있는 픽셀끼리 정규화시켜줍니다. 여러 feature map에서의 결과를 (local)normalization을 시키면, 생물학적 뉴런에서의 lateral inhibition(측면 억제: 강한 자극이 주변의 약한 자극이 전달되는 것을 막는 효과)과 같은 효과를 얻을 수 있기 때문에 generalization 관점에서는 훨씬 좋아지게 됩니다.


특정 layer에 normalization를 적용한 후 ReLU nonlinearity를 적용했으며, Response normalization는 top-1 및 top-5 error rates을 각각 1.4% 및 1.2% 감소시켰습니다. 또한 CIFAR-10 데이터 세트에서 이 체계의 효율성을 확인했는데, 4-layer CNN은 정규화 없이 13%의 test error rate을 달성하고 정규화를 통해 11%를 달성했습니다.


$(x, y)$ position에 있는 $i$번째 kernel에 적용한 다음 ReLU nonlinearity을 적용하여 계산된 뉴런의 activity을 $a^i_{x,y}$로 표시하면, response-normalized activity $b^i_{x,y}$는 다음 식으로 주어집니다.

<p align="center">
<img width="1297" alt="image" src="https://user-images.githubusercontent.com/77891754/179172083-bbf24ab1-6191-4737-a961-ed2ccadcc56c.png" style="zoom:40%;">
</p>

- $a$ : activity of a neuron, kernsl의 i번째 channel에 있는 (x, y)점에서 값
- $b$ : normalized activity, LRN을 적용한 결과 값
- $i$ : i번째 kernel, 현재 Filter
- $(x, y)$ : position
- $n$ : adjacent kernel maps at the same spatial position
- $N$ : total number of kernels in the layer
- $k=2$, $n=5$, $\alpha=1e-4$, $beta=0.75$ : hyper-parameters



Local Response Normalization은 아래 그림의 예를 통해 이해할 수 있습니다. 각각의 다른 색상은 다른 채널을 나타내므로 $N=4$이며, 하이퍼파라미터를 $(k,α,β,n)=(0,1,1,2)$라고 정의하겠습니다. $n=2$라는 의미는 위치 $(i,x,y)$에서 정규화된 값을 계산하는 동안, 이전 및 다음 필터인 $(i-1, x, y)$ and $(i+1, x, y)$에 대해 동일한 위치의 값을 고려한다는 의미입니다. $(i,x,y)=(0,0,0)$의 경우 $normalized\ value(i,x,y) = 1/(1^2+1^2) = 0.5$ 입니다. 나머지 정규화된 값도 비슷한 방식으로 계산됩니다.


<p align="center">
<img width="831" alt="image" src="https://user-images.githubusercontent.com/77891754/179398683-a0b59020-3446-4060-a8f4-1dde0990c5d5.png" style="zoom:40%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : medium</p>




## Overlapping Pooling

<p align="center">
<img width="856" alt="image" src="https://user-images.githubusercontent.com/77891754/179399630-73e2f6b2-f85f-4212-bf15-a35b2853fabe.png" style="zoom:80%;">
</p>

Max Pooling 레이어는 일반적으로 깊이를 동일하게 유지하면서 텐서의 너비와 높이를 다운샘플링하는 데 사용됩니다. Overlapping Max Pooling layer는 일반적인 Max Pooling layer와 유사하지만 최대값이 계산되는 인접한 windows은 서로 겹칩니다.


논문의 저자는 인접한 windows 사이에 stride=2, pooling size=3x3으로 설정하여 겹치는 뉴런을 발생시켰고, 그 결과로 top-1 error rate와 top-5 error rate가 각각 0.4%, 0.3% 감소했습니다. 저자들은 overlapping pooling이 있는 모델이 overfit(과적합)하기가 약간 더 어렵다는 것을 학습하는 동안 관찰했습니다.




## Overall Architecture

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/179400252-5932b46d-d856-4ff7-a1eb-802538eb5e8f.png">
</p>

<p align="center" style="font-size:80%">
Figure 2: 두 GPU 간의 책임 묘사를 명시적으로 보여주는 CNN architecture의 그림. 하나의 GPU는 그림 상단에서 레이어 부분을 실행하고 다른 GPU는 하단에서 레이어 부분을 실행합니다. GPU는 특정 계층에서만 통신합니다. 네트워크의 입력은 150, 528차원이고 네트워크의 나머지 레이어의 뉴런 수는 253,440 – 186,624 – 64,896 – 64,896 – 43,264 – 4096 – 4096 – 1000입니다.
</p>


network에는 가중치가 있는 8개의 layer가 있는데, 처음 5개는 convolutional layer이고 나머지 3개는 fully-connected layer 입니다. 그리고 마지막 fully-connected layer의 출력은 1000개의 클래스에 대한 분포를 생성하는 1000-way softmax에 전달됩니다. network는 multinomial logistic regression objective를 최대화 하는데, 이는 예측 분포에서 올바른 레이블의 log-probability의 training cases의 전반에 걸쳐 평균을 최대화하는 것과 같습니다.


- 두 번째, 네 번째, 다섯 번째 convolutional layers의 kernel은 동일한 GPU에 있는 이전 layer의 kernel map에서만 연결됩니다.
- 세 번째 convolutional layer의 kernel은 두 번째 layer의 모든 kernel map에 연결됩니다.
- fully-connected layers의 뉴런은 이전 layer의 모든 뉴런에 연결됩니다. 
- Response-normalization layers은 첫 번째 및 두 번째 convolutional layer을 따릅니다.
- Max-pooling layers은 다섯 번째 convolutional layer뿐만 아니라 Response-normalization layers을 뒤따릅니다.
- ReLU non-linearity은 모든 convolutional 및 fully-connected layers의 출력에 적용됩니다.


조금 더 자세한 그림을 보겠습니다.


<p align="center">
<img width="1430" alt="image" src="https://user-images.githubusercontent.com/77891754/179400164-d6f15758-8167-4611-8ee4-cf6a82e8a8d5.png">
</p>

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</a></p>


- 첫 번째 convolutional layers는 4픽셀의 stride(이것은 커널 맵에서 인접한 뉴런의 receptive field(수용장) 중심 사이의 거리입니다)로 11 x 11 x 3 크기의 96개 kernel로 227 x 227 x 3 입력 이미지를 필터링합니다.
- 두 번째 convolutional layer는 첫 번째 convolutional layer의(response-normalized and pooled) 출력을 입력으로 받아 5 × 5 × 96 크기의 kernel 256개로 필터링합니다.
- 세 번째 convolutional layer에는 두 번째 convolutional layer의 (normalized, pooled) 출력에 연결된 3 × 3 × 256 크기의 384개의 kernel이 있습니다.
- 네 번째 convolutional layer에는 3 × 3 × 384 크기의 커널이 384개 있습니다.
- 다섯 번째 convolutional layer에는 3 × 3 × 384 크기의 커널이 256개 있습니다.
- fully-connected layers에는 각각 4096개의 뉴런이 있습니다.


이와 관련된 또다른 이미지 입니다.


<p align="center">
<img width="747" alt="image" src="https://user-images.githubusercontent.com/77891754/179400971-d3e20a9d-8359-4480-b9c2-115270b3cec2.png" style="zoom:70%;">
</p>


정리하면 다음과 같습니다.

- Layer 0: Input image
    - Size: 227 x 227 x 3
- Layer 1: Convolution with 96 filters, size 11×11, stride 4, padding 0
    - Size: 55 x 55 x 96
    - (227-11)/4 + 1 = 55 는 결과 사이즈
- Layer 2: Max-Pooling with 3×3 filter, stride 2
    - Size: 27 x 27 x 96
    - (55 – 3)/2 + 1 = 27 는 결과 사이즈
- Layer 3: Convolution with 256 filters, size 5×5, stride 1, padding 2
    - Size: 27 x 27 x 256
    - (5-1)/2=2, 패딩으로 인해 원래 크기로 복원
- Layer 4: Max-Pooling with 3×3 filter, stride 2
    - Size: 13 x 13 x 256
    - (27 – 3)/2 + 1 = 13 는 결과 사이즈
- Layer 5: Convolution with 384 filters, size 3×3, stride 1, padding 1
    - Size: 13 x 13 x 384
    - (3-1)/2=1, 패딩으로 인해 원래 크기로 복원
- Layer 6: Convolution with 384 filters, size 3×3, stride 1, padding 1
    - Size: 13 x 13 x 384
    - (3-1)/2=1, 패딩으로 인해 원래 크기로 복원
- Layer 7: Convolution with 256 filters, size 3×3, stride 1, padding 1
    - Size: 13 x 13 x 256
    - (3-1)/2=1,  패딩으로 인해 원래 크기로 복원
- Layer 8: Max-Pooling with 3×3 filter, stride 2
    - Size: 6 x 6 x 256
    - (13 – 3)/2 + 1 = 6 는 결과 사이즈
- Layer 9: Fully Connected with 4096 neuron
- Layer 10: Fully Connected with 4096 neuron
- Layer 11: Fully Connected with 1000 neurons





# Reducing Overfitting



## Data Augmentation

논문의 저자는 Augmentation이 없었다면 상당한 overfitting에 빠졌을 것이라고 말합니다.



### image horizontal reflections image translation

- 좌우반전(horizontal reflections)을 이용하여 패치를 추출하고, 이미지의 양을 2배로 증가


<p align="center">
<img width="697" alt="image" src="https://user-images.githubusercontent.com/77891754/179401752-c9a6f452-aeea-4bb0-9a56-61c38d0ed3e1.png" style="zoom:70%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</p>



### image translation

- 256×256 이미지에서 랜덤하게 227×227 패치를 추출하여 1024배 증가, 총 2048배 증가


<p align="center">
<img width="726" alt="image" src="https://user-images.githubusercontent.com/77891754/179401759-e874be86-46e1-4ed7-8651-cda47610dc65.png" style="zoom:70%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</p>



### Test

- 5개의 224 × 224 patches(4개의 corner patches 및 1개의 center patch)와 horizontal reflections(따라서 모두 10개의 patches)를 생성
- 10개의 patches에 대해 network의 softmax layer에 의해 만들어진 예측을 평균화함으로써 예측



### jittering

- the top-1 error rate을 1% 이상 감소시킵니다.
- overfitting을 감소시켰습니다.
- training 이미지에서 RGB 채널의 강도를 변경합니다.
- 이미지의 각 RGB 픽셀에 PCA를 적용하여 평균=0, 표준편차=0.1을 갖는 랜덤 변수를 곱한 뒤 기존 픽셀에 더해줍니다.


다음과 같은 값을 모든 픽셀에 더해줍니다.


$$
I_{xy} = [I^R_{xy}, I^G_{xy}, I^B_{xy}]^T +[p_1, p_2, p_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T \\ \alpha_i \sim N(0, 0.1)
$$


- $I_{xy}$는 RGB 이미지 픽셀
- $p_i$ 및 $λ_i$는 각각 RGB 픽셀 값의 3 × 3 공분산 행렬의 $i$번째 고유벡터 및 고유값
- $α_i$는 앞서 언급한 랜덤 변수


각 $α_i$는 해당 이미지가 다시 훈련에 사용될 때까지 특정 training 이미지의 모든 픽셀에 대해 한 번만 그려지고, 그 시점에서 다시 그려집니다. 이 체계는 대략적으로 natural images의 중요한 특성을 포착합니다. 즉, object의 정체성은 조명의 강도와 색상의 변화에 따라 변하지 않는다는 것입니다.





## Dropout

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/179402669-11336bd6-a46a-49b8-b286-c487dda46244.gif" style="zoom:100%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : medium</p>


다양한 모델의 예측을 결합하는 것은 test errors를 줄이는 nice한 방법이지만, 훈련하는 데 며칠이 걸리는 대규모 신경망 네트워크는 깊고 훈련하는데 오래걸리기 때문에 이 앙상블 기법을 사용하기 어려웠습니다. 그러나 훈련하는 동안 비용이 약 2배에 불과한 매우 효율적인 모델인 Dropout을 사용하면 됩니다.


"dropped out"된 뉴런은 순방향 전달에 기여하지 않고 역전파에 참여하지 않기 때문에 입력이 제공될 때마다 신경망은 다른 아키텍처를 샘플링하지만, 이러한 모든 아키텍처는 가중치를 공유합니다. 즉 뉴런들 사이의 의존성을 낮추며, co-adaptations을 감소시킵니다. 따라서 다른 뉴런의 다양한 무작위 하위 집합의 연결이 유용한 보다 더 강건한 feature를 학습하는데 집중하게합니다.


- 0.5의 확률로 각 hidden neuron의 값을 0으로 바꿔줍니다.
- 3개 중 처음 2개의 fully-connected layers에서 dropout을 사용합니다.
- dropout이 없으면 네트워크는 상당한 overfitting을 나타냅니다.
- Test시 dropout 적용 x, 대신 출력에 0.5를 곱해준다.





# Details of learning



## stochastic gradient descent
- batch size : 128
- momentum : 0.9
- weight decay : 0.0005
    - 소량의 가중치 감소가 모델이 학습하는 데 중요하다는 것을 발견했는데, 여기서 가중치 감소는 단순한 정규화가 아니라 모델의 training error를 줄입니다.
- learning rate : 0.01
    - validation error rate가 현재 learning rate로 개선되지 않을 때 learning rate을 10으로 나눔
    - 실험 중 총 3번


가중치 w에 대한 업데이트 규칙은 다음과 같습니다.


<p align="center">
<img width="1010" alt="image" src="https://user-images.githubusercontent.com/77891754/179429810-7e7dcaaa-c6d2-430d-9517-f6b511dfd61a.png" style="zoom:50%;">
</p>


- $i$ : iteration index
- $v$ : momentum variable
- $ε$ : learning rate,
- $\left\langle \frac{∂L}{∂w}|_{w_i} \right\rangle_{D_i}$ : $w_i$에서 평가된, $w$에 대한 목적 도함수의 $i$번째 배치 $D_i$에 대한 평균입니다.



## weight initialization
- mean=0, std=0.01 Gaussian distribution 초기화
- 첫 번째, 세 번째 convolutional layers의 biases : 상수 0으로 초기화
- 나머지 layer : 상수 1로 초기화
    - 이 초기화는 ReLU에 positive인 inputs을 제공하여 학습의 초기 단계를 가속화시킵니다.



## train

- dropout : 0.5
- epoch : 90
- 2개의 NVIDIA GTX 580 3GB GPU에서 5~6일 소요





# Results

ILSVRC-2010에 대한 결과는 Table 1에 요약되어 있습니다.


<p align="center">
<img width="1441" alt="image" src="https://user-images.githubusercontent.com/77891754/179430873-aa2937c8-cee2-4aff-8176-95df20dd77a4.png" style="zoom:30%;">
</p>


본 네트워크는 top-1 및 top-5 test set error rates를 각각 37.5%와 17.0%의 달성하였습니다. 당시 발표된 방법 중 47.1%, 28.2%를 기록한 팀은 6개의 sparse-coding된 모델의 예측을 평균하여 기록을 달성하였고, 45.7%, 25.7%의 모델은 2가지 종류의 밀접한 특성을 이용한 Fisher Vectors를 계산하여 2개의 분류기의 예측을 평균하여 기록을 내었는데, 당시의 방법론 보다 훨씬 좋은 성능을 보여주었습니다.


또한 ILSVRC-2012에도 참가하였는데 그에 대한 기록은 Table 2에 나와있습니다.


<p align="center">
<img width="1670" alt="image" src="https://user-images.githubusercontent.com/77891754/179430888-2609a1fc-c824-4035-b0ec-5aca1f0232cb.png" style="zoom:35%;">
</p>


ILSVRC-2012 test set labels은 라벨링이 되어있지 않아서 우리가 시도한 모든 모델들의 test error rates을 기록하지는 못했습니다. 특히 val 과 test의 error rates이 0.1%의 이상 차이가 나지 않기 때문에 둘을 같은 결과로 사용했습니다.


- 본 논문에서의 CNN은 18.2%의 top-5 error rate을 기록했습니다.
- 비슷한 CNN 5개의 예측을 평균하면 16.4%의 error rate가 나타납니다.
- 마지막 pooling layer 위에 여섯 번째 convolutional layer가 추가된 하나의 CNN을 training하여, 전체 ImageNet Fall 2011 release를 분류한 다음, ILSVRC-2012에서 "fine-tuning"하면 error rate가 16.6%에 달합니다.
- 앞서 언급한 fine-tuning 모댈 2개와 5개의 CNN을 예측을 평균하면 15.3%의 error rate을 얻을 수 있습니다.





# Qualitative Evaluations


<p align="center">
<img width="1720" alt="image" src="https://user-images.githubusercontent.com/77891754/179430570-8c42826a-3bcc-410a-9d56-ade3ec382c38.png" style="zoom:40%;">
</p>

<p align="center" style="font-size:80%">Figure 3: 224x224x3 입력 이미지에서 첫 번째 convolutional layer에서 학습한 11x11x3 크기의 96개 convolutional kernel. 상위 48개 kernel은 GPU 1에서 학습되고 하위 48개 kernel은 GPU 2에서 학습되었습니다.</p>


이 network에서는 다양한 주파수(frequency), 방향(orientation-selective), 색상(blobs)들을 학습했습니다. GPU1의 kernels은 주로 색깔정보가 없지만 GPU2의 kernels은 다양한 색상을 담고있습니다. 이런 특성은 랜덤한 가중치 초기화와는 무관하게 매 실행마다 발생합니다.


<p align="center">
<img width="1728" alt="image" src="https://user-images.githubusercontent.com/77891754/179430613-ac5992a8-2b39-46bb-b180-4e0744a853a0.png" style="zoom:100%;">
</p>

<p align="center" style="font-size:80%">(Left) 8개의 ILSVRC-2010 테스트 이미지와 우리 모델에서 가장 가능성이 높은 것으로 간주되는 5개의 레이블. 올바른 레이블이 각 이미지 아래에 기록되고 올바른 레이블에 할당된 확률도 빨간색 막대로 표시됩니다(상위 5개에 있는 경우). (Right) 첫 번째 열에 있는 5개의 ILSVRC-2010 테스트 이미지. 나머지 열은 테스트 이미지에 대한 특징 벡터로부터 유클리드 거리가 가장 작은 마지막 은닉층에서 특징 벡터를 생성하는 6개의 훈련 이미지를 보여줍니다.</p>


Figure 4의 왼쪽 panel에서 network가 8개의 test images에 대한 top-5 예측을 계산하여 학습한 내용을 정성적으로 평가합니다. 또한 왼쪽 상단의 진드기와 같이 중심에서 벗어난 objects도 network에 의해 인식될 수 있으며, top-5 레이블의 대부분은 합리적으로 보입니다. 예를들어 고양이종에 속하는 다른 것들이 표범의 하위 라벨들로 예측되었으며, 아래의 첫 번째와 세 번째 이미지인 cherry나 grille의 경우 초점을 어디에 맞추느냐에 따라 충분히 나올 수 있는 답이기 때문에 완전한 오류라고 볼 수는 없습니다.


신경망의 시각적인 지식(visual knowledge)을 확인하는 또다른 방법은 마지막 4096개의 은닉층에서 이미지에 의해 유도된 특성 활성화(feature activations)입니다. 만약 두 이미지가 작은 Euclidean separation으로 특성이 활성화 되었다면, 우리는 신경망이 매우 높은 수준(high levels)으로 그 둘이 비슷하다고 생각합니다. Figure4의 오른쪽 panel에서 이 방법에 따르면 테스트셋의 5개의 이미지와 6개의 훈련이미지가 매우 비슷하다는 것을 보여줍니다. 픽셀 수준에서 생각하면, 검색된 훈련 이미지는 첫번째 열의 이미지와 L2에서 비슷하지 않습니다.(색이나 포즈 등) 그럼에도 불구하고 같은 부류라고 판단하는 것을 볼 수 있습니다. 예를 들어, 개와 코끼리 사진은 다양한 포즈를 취합니다.


이를 통해 단순히 픽셀이 아닌, 더 고차원적인 근거로 분류한다는 것을 알 수 있습니다.




# Discussion

본 논문의 결과는 크고 깊은 convolutional neural network이 순수 supervised learning을 사용하여 매우 까다로운 dataset에서 기록적인 결과를 달성할 수 있음을 보여줍니다. 그리고 convolutional layer 중 하나라도 없앨 경우 성능이 크게 떨어지므로 것을 이유로 모델의 깊이가 그만큼 중요합니다. 또한 학습 이전에 비지도학습으로 미리 학습을 했더라면(unsupervised pre-training) 성능이 더 좋았을 것이라고 가정하고 있습니다.


최종적으로 지금까지의 결과는 네트워크를 더 크게 만들고 더 오래 훈련했기 때문에 개선되었지만, human visual system의 infero-temporal pathway와 일치시키기 위해서는 아직 많은 노력이 필요합니다. 궁극적으로 우리는 정적이거나 일시적인 형태에서 매우 많은 정보를 주는 비디오 시퀀스에서 크고 넓은 CNN을 적용하고 싶다고 합니다.




