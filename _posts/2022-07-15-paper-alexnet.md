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

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</a></p>


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



Local Response Normalization은 아래 그림의 예를 통해 이해할 수 있습니다. 각각의 다른 색상은 다른 채널을 나타내므로 $N=4$이며, 하이퍼파라미터를 $(k,α,β,n)=(0,1,1,2)$라고 정의하겠습니다. $n=2$라는 의미는 위치 $(i,x,y)$에서 정규화된 값을 계산하는 동안, 이전 및 다음 필터인 $(i-1, x, y)$ and $(i+1, x, y)$에 대해 동일한 위치의 값을 고려한다는 의미입니다. $(i,x,y)=(0,0,0)$의 경우 $normalized_value(i,x,y) = 1/(1^2+1^2) = 0.5$ 입니다. 나머지 정규화된 값도 비슷한 방식으로 계산됩니다.


<p align="center">
<img width="831" alt="image" src="https://user-images.githubusercontent.com/77891754/179398683-a0b59020-3446-4060-a8f4-1dde0990c5d5.png" style="zoom:40%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : medium</a></p>




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

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</a></p>



### image translation

- 256×256 이미지에서 랜덤하게 227×227 패치를 추출하여 1024배 증가, 총 2048배 증가


<p align="center">
<img width="726" alt="image" src="https://user-images.githubusercontent.com/77891754/179401759-e874be86-46e1-4ed7-8651-cda47610dc65.png" style="zoom:70%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : learnopencv</a></p>



### Test

- 5개의 224 × 224 patches(4개의 corner patches 및 1개의 center patch)와 horizontal reflections(따라서 모두 10개의 patches)를 생성
- 10개의 patches에 대해 network의 softmax layer에 의해 만들어진 예측을 평균화함으로써 예측



### jittering

- the top-1 error rate을 1% 이상 감소시킵니다.
- overfitting을 감소시켰습니다.
- training 이미지에서 RGB 채널의 강도를 변경합니다.
- RGB 픽셀 값 세트에 대해 PCA를 수행하여 발견된 주성분의 배수를 추가해줍니다.


다음과 같은 값을 모든 픽셀에 더해줍니다.


$$
I_{xy} = [I^R_{xy}, I^G_{xy}, I^B_{xy}]^T +[p_1, p_2, p_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T \\ \alpha_i ~ N(0, 0.1)
$$


여기서 $I_{xy}$는 RGB 이미지 픽셀이며, $p_i$ 및 $λ_i$는 각각 RGB 픽셀 값의 3 × 3 공분산 행렬의 $i$번째 고유벡터 및 고유값이고, $α_i$는 앞서 언급한 랜덤 변수입니다. 각 $α_i$는 해당 이미지가 다시 훈련에 사용될 때까지 특정 training 이미지의 모든 픽셀에 대해 한 번만 그려지고, 그 시점에서 다시 그려집니다. 이 체계는 대략적으로 natural images의 중요한 특성을 포착합니다. 즉, object의 정체성은 조명의 강도와 색상의 변화에 따라 변하지 않는다는 것입니다.





## Dropout

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/179402669-11336bd6-a46a-49b8-b286-c487dda46244.gif" style="zoom:100%;">
</p>

<p align="center" style="font-size:80%">이미지 출처 : medium</a></p>


다양한 모델의 예측을 결합하는 것은 test errors를 줄이는 nice한 방법이지만, 훈련하는 데 며칠이 걸리는 대규모 신경망에는 너무 많은 비용이 듭니다. 그러나 훈련하는 동안 비용이 약 2배에 불과한 매우 효율적인 모델인 Dropout을 사용하면 됩니다.


"dropped out"된 뉴런은 순방향 전달에 기여하지 않고 역전파에 참여하지 않기 때문에 입력이 제공될 때마다 신경망은 다른 아키텍처를 샘플링하지만, 이러한 모든 아키텍처는 가중치를 공유합니다. 즉 뉴런들 사이의 의존성을 낮추며, co-adaptations을 감소시킵니다. 따라서 다른 뉴런의 다양한 무작위 하위 집합의 연결이 유용한 보다 더 강건한 feature를 학습하는데 집중하게합니다.


- 0.5의 확률로 각 hidden neuron의 값을 0으로 바꿔줍니다.
- 3개 중 처음 2개의 fully-connected layers에서 dropout을 사용합니다.
- dropout이 없으면 네트워크는 상당한 overfitting을 나타냅니다.
- Test시 dropout 적용 x, 대신 출력에 0.5를 곱해준다.





# Details of learning

우리는 128개의 batch size, 0.9의 momentum, 0.0005의 weight decay를 사용하여 stochastic gradient descent을 사용하여 모델을 훈련했습니다. 우리는 이 소량의 가중치 감소가 모델이 학습하는 데 중요하다는 것을 발견했습니다. 다시 말해, 여기서 가중치 감소는 단순한 정규화가 아니라 모델의 training error를 줄입니다.

가중치 w에 대한 업데이트 규칙은 다음과 같습니다.

- $i$ : iteration index
- $v$ : momentum variable
- $ε$ : learning rate,
- $\left\langle \frac{∂L}{∂w}|_{w_i} \right\rangle_{D_i}$ : $w_i$에서 평가된, $w$에 대한 목적 도함수의 $i$번째 배치 $D_i$에 대한 평균입니다.


표준 편차가 0.01인 zero-mean Gaussian distribution에서 각 레이어의 가중치를 초기화했습니다. 두 번째, 네 번째, 다섯 번째 convolutional layers와 fully-connected hidden layers에서 neuron biases을 상수 1로 초기화했습니다. 이 초기화는 ReLU에 positive인 inputs을 제공하여 학습의 초기 단계를 가속화시킵니다. 나머지 layers의 neuron biases을 상수 0으로 초기화했습니다.


우리는 모든 layers에 대해 동일한 learning rate을 사용했으며, training 내내 수동으로 조정했습니다. 우리가 따랐던 휴리스틱은 validation error rate가 현재 learning rate로 개선되지 않을 때 learning rate을 10으로 나누는 것이었습니다. learning rate은 0.01로 초기화되었고 종료 전에 3번 감소했습니다. 우리는 120만개의 이미지인 training set가 대략 90번의 cycles을 하는 네트워크를 훈련했고, 2개의 NVIDIA GTX 580 3GB GPU에서 5~6일이 소요되었습니다.




# Results

ILSVRC-2010에 대한 결과는 표 1에 요약되어 있습니다. 우리의 네트워크는 37.5%와 17.0%의 top-1 and top-5 test set error rates을 달성했습니다. ILSVRC 2010 대회에서 달성한 최고의 성능은 서로 다른 feature에 대해 training된 6개의 sparse-coding 모델에서 생성된 예측을 평균화하는 접근 방식으로 47.1%와 28.2%였으며, 그러고 나서 가장 잘 발표된 결과는 두 가지 유형의 조밀하게 샘플링된 features로 부터 계산되는 Fisher Vectors(FVs)에 대해 훈련된 두 분류기의 예측을 평균화하는 접근 방식으로 45.7%와 25.7%였다.


우리는 또한 ILSVRC-2012 대회에 우리 모델을 입력했고 결과를 표 2에 보고했습니다. ILSVRC-2012 test set labels은 공개적으로 사용할 수 없기 때문에, 우리가 시도한 모든 모델에 대한 test error rates를 보고할 수 없습니다. 이 단락의 나머지 부분에서는 경험상 0.1% 이상 차이가 나지 않기 때문에 validation and test error rates을 서로 바꿔서 사용합니다(표 2 참조). 이 논문에서 설명하는 CNN은 18.2%의 top-5 error rate을 달성했습니다. 5개의 유사한 CNN의 예측을 평균하면 16.4%의 error rate이 나타납니다. 마지막 pooling layer 위에 여섯 번째 convolutional layer가 추가된 하나의 CNN을 training하여, 전체 ImageNet Fall 2011 release(15M images, 22K categories)를 분류한 다음, ILSVRC-2012에서 "fine-tuning"하면 error rate가 16.6%에 달합니다. 앞서 언급한 5개의 CNN을 사용하여 전체 2011년 Fall release에서 pre-trained된 2개의 CNN 예측을 평균하면 15.3%의 error rate을 얻을 수 있습니다. 두 번째로 우수한 공모전 출품작은 서로 다른 유형의 조밀하게 샘플링된 feature에서 계산된 FV에 대해 훈련된 여러 분류기의 예측을 평균화하는 접근 방식으로 26.2%의 error rate을 달성했습니다.


마지막으로 10,184개의 카테고리와 890만 개의 이미지가 있는 ImageNet의 2009년 Fall 버전에 대한 error rate도 보고합니다. 이 dataset에서 우리는 이미지의 절반을 training에 사용하고 절반을 test에 사용하는 문헌의 규칙을 따릅니다. 확정된 test set가 없기 때문에, 우리의 분할은 이전 작성자가 사용한 분할과 다르지만, 결과에 상당한 영향은 미치지 않습니다. 이 dataset의 top-1 및 top-5 error rates은 67.4% 및 40.9%로, 위에서 설명한 네트워크에 의해 달성되지만 마지막 풀링 계층 위 추가로 여섯 번째 컨볼루션 계층이 있습니다. 이 dataset에서 가장 잘 발표된 결과는 78.1%와 60.9%입니다.






# Qualitative Evaluations



그림 3은 network의 2개의 data-connected layers에서 학습한 convolutional kernels을 보여줍니다. 네트워크는 다양한 색상의 blobs뿐만 아니라 다양한 frequency 및 orientation-selective kernels을 학습했습니다. 섹션 3.5에 설명된 제한된 연결의 결과인 두 GPU에 의해 나타나는 specialization에 주목하십시오. GPU 1의 커널은 대체로 색상에 구애받지 않는 반면, GPU 2의 커널은 대부분 색상에 따라 다릅니다. 이러한 종류의 specialization는 모든 실행 중에 발생하며, 특정 랜덤 가중치 초기화(modulo a renumbering of the GPUs)와는 무관합니다.


그림 4의 왼쪽 panel에서 우리는 network가 8개의 test images에 대한 top-5 예측을 계산하여 학습한 내용을 정성적으로 평가합니다. 왼쪽 상단의 진드기와 같이 중심에서 벗어난 objects도 network에 의해 인식될 수 있습니다. top-5 레이블의 대부분은 합리적으로 보입니다. 예를 들어, 다른 종류의 고양이만 표범에 대한 그럴듯한 라벨로 간주됩니다. 경우에 따라(grille, cherry) 사진의 의도된 초점이 모호합니다.



network의 visual knowledge을 조사하는 또 다른 방법은 마지막 4096-dimensional hidden layer에서 이미지에 의해 유도된 feature activations를 고려하는 것입니다.

두 이미지가 작은 Euclidean 분리로 feature activation vectors를 생성한다면, 우리는 신경망의 더 높은 levels이 그들과 유사한 것으로 간주한다고 말할 수 있습니다.(???)

그림 4는 이 측정에 따라 test set의 5개 이미지와 각각 가장 유사한training set의 6개 이미지를 보여줍니다.

픽셀 수준에서 검색된 training images는 일반적으로 첫 번째 열의 쿼리 이미지와 L2에서 가깝지 않습니다.

예를 들어, 검색된 개와 코끼리는 다양한 포즈를 취합니다. 우리는 보충 자료에 더 많은 테스트 이미지에 대한 결과를 제시합니다.







두 4096차원 실제 값 벡터 사이의 유클리드 거리를 사용하여 유사성을 계산하는 것은 비효율적이지만, 이러한 벡터를 짧은 이진 코드로 압축하도록 auto-encoder를 training 함으로써 효율적일 수 있습니다.

이것은 이미지 레이블을 사용하지 않기 때문에 의미적으로 유사하든 그렇지 않든 유사한 가장자리의 패턴을 가진 이미지를 검색하는 경향이 있는, 원시 픽셀에 auto-encoder를 적용하는 것보다 훨씬 더 나은 이미지 검색 방법을 생성해야 합니다.

<p align="center">
<img width="1740" alt="image" src="https://user-images.githubusercontent.com/77891754/179342930-d4248786-8375-403b-b3ef-fbebf1f17ad9.png">
</p>

<p align="center" style="font-size:80%">
(Left) 8개의 ILSVRC-2010 테스트 이미지와 우리 모델에서 가장 가능성이 높은 것으로 간주되는 5개의 레이블. 올바른 레이블이 각 이미지 아래에 기록되고 올바른 레이블에 할당된 확률도 빨간색 막대로 표시됩니다(상위 5개에 있는 경우).

(Right) 첫 번째 열에 있는 5개의 ILSVRC-2010 테스트 이미지. 나머지 열은 테스트 이미지에 대한 특징 벡터로부터 유클리드 거리가 가장 작은 마지막 은닉층에서 특징 벡터를 생성하는 6개의 훈련 이미지를 보여줍니다.
</p>

# Discussion



우리의 결과는 크고 깊은 convolutional neural network이 순수 supervised learning을 사용하여 매우 까다로운 dataset에서 기록적인 결과를 달성할 수 있음을 보여줍니다. 단일 convolutional layer가 제거되면 네트워크 성능이 저하된다는 점은 주목할 만합니다. 예를 들어 중간 layer을 제거하면 network의 top-1 performance에 대해 약 2%의 손실이 발생합니다. 따라서 깊이는 결과를 달성하는 데 정말 중요합니다.



실험을 단순화하기 위해, 특히 라벨링 된 데이터의 양에 상응하는 증가 없이 네트워크 크기를 크게 늘릴 수 있는 충분한 계산 능력을 얻는 경우, 도움이 될 것으로 예상되지만 unsupervised pre-training을 사용하지 않았습니다. 지금까지 우리의 결과는 네트워크를 더 크게 만들고 더 오래 훈련했기 때문에 개선되었지만, human visual system의 infero-temporal pathway와 일치시키기 위해서는 아직 가야 할 많은 양의 순서가 있습니다. 궁극적으로 우리는 temporal structure가 static images에서 누락되거나 훨씬 덜 분명하지만 매우 유용한 정보를 제공하는 video sequences에 매우 크고 깊은 convolutional nets를 사용하고 싶습니다.



