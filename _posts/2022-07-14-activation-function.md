---
1layout: post
title: 활성화 함수(Activation)와 비선형 함수(Non-linear function)
category: Deep Learning
tag: Deep-Learning
---




# Activation function

오늘은 수학 분야에서도 딥러닝과 아주아주 밀접하고 직접적인 주제를 다루어보겠습니다. 바로 softmax나 ReLU 등 이미 익숙히 들어보셨을 **활성화 함수(activation function)** 입니다. 활성화 함수란 무엇일까요? "어떤 것이 활성화(activated)되었다" 라는 것을 들으면 어떤 것이 떠오르시나요? **활성화(activated)** or **비활성화(deactivated)** 라는 것은 '어떤 조건을 만족 or 불만족했다'라는 것과 긴밀한 연관이 있습니다.


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178718846-4f8c3f43-f76f-4676-b3c6-8306295f8002.png" alt="activation f" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.v7labs.com/blog/neural-networks-activation-functions">v7labs</a></p>


즉 신경망 속의 퍼셉트론(perceptron) 혹은 노드(node)는 '특정 조건'이 만족하면 '활성화' 되도록 디자인되어 있으며, 노드에 입력으로 들어오는 값이 어떤 '임계치'를 넘어가면 "활성화(activated)"되고, 넘어가지 않으면 "비활성화(deactivated)"되게끔 설계되어 있습니다. 즉 계단 함수(Step function)를 통해 출력값이 0이 될지, 1이 될지를 결정했습니다. 이러한 매커니즘은 실제 뇌를 구성하는 신경 세포 뉴런이 전위가 일정치 이상이 되면 시냅스가 서로 화학적으로 연결되는 모습을 모방한 것입니다. 이렇게 **은닉층과 출력층의 뉴런에서 출력값을 결정하는 함수** 를 **활성화 함수(Activation function)** 라고 합니다.


활성화 함수의 기본적 정의는 위와 같지만, 실제로 딥러닝에서 활성화 함수를 쓰는 결정적 이유는 따로 있습니다. 바로 **신경망에 비선형성을 추가하여 딥러닝 모델의 표현력을 향상**시켜주기 위해서인데요, 전문적인 용어로는 모델의 **representation capacity** 또는 **expressivity** 를 향상시킨다라고도 말합니다.



## improve expressivity

활성화 함수는 모델의 표현력을 왜 향상시켜줄까요? 답은 간단합니다. 만일 어떤 모델이 $w_1, b_1$이라는 2개의 parameter로 이루어진 다음과 같은 모델이라고 해보겠습니다.


$$
f(x)=w_1x+b_1
$$

그런데 이 모델로 $x^2, x^5, sin(x)$등으로 표현되는 데이터를 학습할 수 있을까요? 답은 "그럴 수 없다" 입니다. 왜냐하면 $w_1, b_1$값을 아무리 바꿔도 $x^2, x^5, sin(x)$와 같은 함수는 절대 표현할 수 없기 때문이죠. 이를 수학적으로 말하면, **'"선형" 함수(직선)로는 "비선형"함수(사인곡선 or $x^5$와 같은 고차항)fmf 표현할 수 없다'**라고 말합니다.

그런데 잘 생각해 보시면 딥러닝 모델의 parameter($w,b$)들은 입력값 $x$와 선형 관계입니다. 왜냐하면, $wx+b$의 표현되는, 즉 곱하고 더하는 연산만 하면서 그다음 layer로 전달하기 때문이죠. 그리고 아무리 많은 layer들을 겹쳐도 역시 그 결과는 선형 관계입니다. 따라서 사인 곡선처럼 직선으로는 근사 시킬 수 없는 (혹은 고양이나 강아지 사진처럼 무수히 많고 복잡한 특징들을 가진) **비선형 데이터를 표현하려면 딥러닝 모델도 비선형성을 지니고 있어야** 합니다. 이때 쓰인 것이 바로 **활성화 함수**이고, 이 **활성화 함수를 layer 사이사이에 넣어줌으로써 모델이 비선형 데이터도 표현**할 수 있게 되었습니다.




<br>

# Linear and Non-linear


딥러닝에서는 일반적으로 비선형 활성화 함수를 사용한다고 합니다. 그럼 선형 활성화 함수는 왜 딥러닝에서 사용되지 않는 걸까요? 이를 알아보기 위해 먼저 선형(Linear)에 대해 알아보겠습니다.

## Linear