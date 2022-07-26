---
title: Visualizing and Understanding Convolutional Networks
category: Classification Paper
tag: classification-paper
date: 2022-07-15
---     


<p align="center">
<img width="1616" alt="image" src="https://user-images.githubusercontent.com/77891754/179626161-b8f78d03-a3cb-44ce-b843-7c8f7ff7c06d.png">
</p>





# Abstract

Large Convolutional Network model은 최근 ImageNet 벤치마크에서 인상적인 분류 성능을 보여주었습니다(Krizhevsky et al., 2012). 그러나 model이 왜 그렇게 잘 작동하는지, 또는 어떻게 개선될 수 있는지에 대한 명확한 이해가 없습니다.

이 논문에서 우리는 두 가지 문제를 다룹니다. 중간 feature layers의 기능과 classifier의 작동에 대한 insight를 제공하는 새로운 시각화 기법을 소개합니다. 진단 역할에 사용되는 이러한 시각화를 통해 ImageNet classification benchmark에서의 Krizhevsky et al을 능가하는 model architectures를 찾을 수 있습니다. 또한 다양한 model layers에서 성능 기여를 발견하기 위해 ablation study를 수행합니다. 우리는 우리의 ImageNet model이 다른 datasets에 잘 일반화되었음을 보여줍니다. softmax classifier가 retrained될 때, Caltech-101 및 Caltech-256 datasets에 대한 최신 SOTA 결과를 설득력 있게 능가합니다.



# Introduction
## Related Work

# Approach
## Visualization with a Deconvnet

# Training Details

# Convnet Visualization
## Architecture Selection
## Occlusion Sensitivity
## Correspondence Analysis

# Experiments

## ImageNet 2012
## Feature Generalization
## Feature Analysis


# Discussion