---
title: Visualizing and Understanding Convolutional Networks
category: Classification Paper
tag: classification-paper
date: 2022-07-15
---     


<p align="center">
<img width="1616" alt="image" src="https://user-images.githubusercontent.com/77891754/182078551-269eff8b-646b-412a-a525-eb8f50a5643f.png">
</p>





# Abstract

Large Convolutional Network model은 ImageNet 벤치마크에서 인상적인 분류 성능을 보여주었습니다. 하지만 model이 왜  잘 작동하는지, 또는 어떻게 개선될 수 있는지에 대한 명확한 이해가 없습니다.

* 본 논문에서 두 가지 문제를 다룹니다.
    * 중간 feature layers의 기능과 classifier의 작동에 대한 insight를 제공하는 새로운 시각화 기법을 소개합니다.
    * 또한 다양한 model layers에서 성능 기여를 발견하기 위해 ablation study를 수행합니다. ZFNet은 다른 datasets에 잘 일반화되었음을 보여줍니다: softmax classifier가 retrained될 때, Caltech-101 및 Caltech-256 datasets에 대한 최신 SOTA 결과를 설득력 있게 능가합니다.





# Introduction

1990년대 초 LeCun에 의해 소개된 이후, Convolutional Networks는 hand-written digit classification과 face detection와 같은 작업에서 우수한 성능을 보여주었습니다. 가장 주목할 만한 것은 Alex Krizhevsky에 의해 ImageNet 2012 classification benchmark에서 기록적인 성능을 보여, 16.4%의 error rate을 달성했다는 것입니다.

ConvNet model에 대한 관심에는 몇 가지 요인이 있습니다.
1. 수백만 개의 라벨링이 된 dataset과 함께, 훨씬 더 큰 training sets의 가용성
2. 매우 큰 model training을 실용적으로 만드는 강력한 GPU 구현
3. Dropout과 같은 더 나은 model regularization 전략

엄청난 발전에도 불구하고, 이런 복잡한 model의 내부 operation 및 behavior, 또는 어떻게 우수한 성능을 달성하는지에 대한 insight는 여전히 거의 없습니다. 과학적 관점에서 이것은 매우 불만족스러우며, 이러한 model이 어떻게 작동하는지와 이유를 명확하게 이해하지 못한다면, 더 나은 모델의 개발은 시행착오에 그치게 됩니다.

본 논문의 핵심 사항은 다음과 같습니다.
* 모델의 모든 layer에서 개별 feature maps을 활성화시키는 입력 자극(input stimuli)을 나타내는 시각화 기술을 소개합니다.
* Zeiler가 제안한 시각화 기술인 multi-layered Deconvolutional Network (deconvnet)를 사용하여 feature activations을 입력 pixel 공간으로 다시 투영하는 방법입니다.
* 이를 통해 훈련중에 features의 발전(evolution)을 관찰하고 모델의 잠재적인 문제를 진단할 수 있습니다.
* 또한 입력 이미지의 일부를 가려서 classifier output의 민감도 분석을 수행하여, scene의 어떤 부분이 분류에 중요한지 나타냅니다.
* 위와 같은 방법으로 다양한 architecture를 탐색하고, ImageNet에서 AlexNet의 결과를 능가하는 architecture를 찾습니다. 이후 softmax classifier를 retraining하여, 다른 dataset에 대한 모델의 일반화 능력을 탐색합니다.



## Related Work

* network에 대한 직관을 얻기 위해 feature를 시각화하는 것은 일반적이지만, 대부분 pixel 공간에 대한 투영이 가능한 첫 번째 layer로 제한된 연구였습니다. 즉 higher layers에서는 그렇지 않으며, activity을 해석하는 방법이 제한적입니다.
    * layer를 시각화 하는 방법 중 하나는 unit의 activation를 maximize하기 위해 이미지 공간에서 gradient descent를 수행하여 각 unit에 대한 최적의 stimuli을 찾는 방법이 있지만, 이것은 careful initialization가 필요하며 unit의 invariances에 대한 정보를 제공받지 못하는 단점이 있습니다.
    * layer를 시각화 하는 또다른 방법으로 주어진 unit의 헤시안(Hesian)이 어떻게 optimal response을 중심으로 수치적으로 계산될 수 있는지 보여줌으로써 invariances에 대한 insight를 제공합니다. 하지만 higher layers의 경우 invariances이 매우 복잡하여 간단한 2차 근사에 의해 잘 포착되지 않는다는 것입니다.
* 이와 대조적으로 본 논문의 접근 방식은 training set에 대해 어떤 패턴이 feature map을 활성시키는지 보기위해, invariance를 non-parametric한 관점에서 제공한 기법을 연구했습니다.
* 또한 우리의 시각화는 input images의 크롭이 아니라, 특정 feature map을 자극하는 각 patch 내의 구조를 드러내는 하향식 투영을 진행합니다.



## Unit's Invariance

패턴인식의 관점에서 Ideal한 feature란 robust하고 selective한 feature를 말합니다. Hidden layer의 Unit들은 feature detector로 해석될 수 있으며, 이는 hidden unit이 표현하는 feature가 현재 입력에 존재하는 경우 strongly respond하고 현재 입력에 부재하는 경우 weakly respond하는 경향을 보이는 것을 의미합니다. 

Invariant 한 뉴런이라는 것은 입력이 특정 변환들을 거치게 되더라도 해당 feature로의 high response를 유지하는 것을 의미합니다. (예를 들면, 어떤 뉴런이 얼굴의 특징을 탐지하는 뉴런이라고 할때, 얼굴이 Rotate되더라도 response를 잘 해내는 것이라고 할 수 있습니다.)





# Approach

본 논문 전반에걸쳐 standard fully supervised convnet models인 AlexNet을 사용합니다. 해당 모델은 일련의 layers를 통해 color 2D input image $x_i$를, $C$개의 다른 클래스에 대한 확률 벡터 $\hat{y}_i$에 매핑(map)합니다.



## Visualization with a Deconvnet


convnet의 작동을 이해하려면 중간 layers의 feature activity을 해석해야 합니다. 이러한 activities을 input pixel space에 다시 매핑(mapping)하는 새로운 방법을 제시하여, feature maps에서 주어진 activation을 원래 어떤 input pattern이 일으켰는지 보여줍니다.

이러한 mapping을 Deconvolutional Network(deconvnet)로 수행합니다. deconvnet은 같은 components(filtering, pooling)을 사용하는 convnet model로 생각할 수 있지만, pixels이 features로 mapping되는 과정의 반대과정을 수행합니다.


<p align="center">
<img width="1616" alt="image" src="https://user-images.githubusercontent.com/77891754/182133084-46771cd7-0d77-4ded-9794-e6e42834fe66.png" style="zoom:40%;">
</p>

<p align="center" style="font-size:80%">Figure 1. Unpooling을 통해 feature map을 image로 reconstruction하는 과정</p>


Convnet을 조사하기 위해, deconvnet은 Figure 1과 같이 각 layers에 부착되어, image pixels로 되돌아오는 연속적인 경로를 제공합니다.
* input image가 convnet과 layers 전체에 걸쳐 계산된 features에 제시됩니다.
* 주어진 convnet activation를 조사하기 위해, layer의 다른 모든 activations를 0으로 설정하고 feature maps을 연결된 deconvnet layer에 대한 입력으로 전달합니다.
* 그런 다음 우리는 연속적으로 (i) unpool, (ii) rectify, (iii) 선택된 activation를 일으킨 아래 layer의 activity을 재구성하기 위해 필터링합니다. 그런 다음 input pixel space에 도달할 때까지 반복됩니다.



















# Training Details

Figure 3은 본 논문의 실험에서 많이 사용된 모델을 보여줍니다.

<p align="center">
<img width="1616" alt="image" src="https://user-images.githubusercontent.com/77891754/182122718-cedf4865-41e4-4ca5-b8aa-fa277533838d.png">
</p>

<p align="center" style="font-size:80%">Figure 3. 8-layer convnet model architecture.</p>





# Convnet Visualization
## Architecture Selection
## Occlusion Sensitivity
## Correspondence Analysis

# Experiments

## ImageNet 2012
## Feature Generalization
## Feature Analysis


# Discussion