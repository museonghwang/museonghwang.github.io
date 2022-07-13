---
layout: post
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

그런데 이 모델로 $x^2, x^5, sin(x)$등으로 표현되는 데이터를 학습할 수 있을까요? 답은 "그럴 수 없다" 입니다. 왜냐하면 $w_1, b_1$값을 아무리 바꿔도 $x^2, x^5, sin(x)$와 같은 함수는 절대 표현할 수 없기 때문이죠. 이를 수학적으로 말하면, **'"선형" 함수(직선)로는 "비선형"함수(사인곡선 or $x^5$와 같은 고차항)를 표현할 수 없다'**라고 말합니다.

그런데 잘 생각해 보시면 딥러닝 모델의 parameter($w,b$)들은 입력값 $x$와 선형 관계입니다. 왜냐하면, $wx+b$의 표현되는, 즉 곱하고 더하는 연산만 하면서 그다음 layer로 전달하기 때문이죠. 그리고 아무리 많은 layer들을 겹쳐도 역시 그 결과는 선형 관계입니다. 따라서 사인 곡선처럼 직선으로는 근사 시킬 수 없는 (혹은 고양이나 강아지 사진처럼 무수히 많고 복잡한 특징들을 가진) **비선형 데이터를 표현하려면 딥러닝 모델도 비선형성을 지니고 있어야** 합니다. 이때 쓰인 것이 바로 **활성화 함수**이고, 이 **활성화 함수를 layer 사이사이에 넣어줌으로써 모델이 비선형 데이터도 표현**할 수 있게 되었습니다.




<br>

# Linear and Non-linear


딥러닝에서는 일반적으로 비선형 활성화 함수를 사용한다고 합니다. 그럼 선형 활성화 함수는 왜 딥러닝에서 사용되지 않는 걸까요? 이를 알아보기 위해 먼저 선형(Linear)에 대해 알아보겠습니다.

## 선형(Linear)

선형 변환이란 '선형'이라는 규칙을 지키며 $V$, 공간상의 벡터를 $W$ 공간상의 벡터로 바꿔주는 역할을 합니다.


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178720377-4ef885f7-9241-4b38-8ad3-b17b113c8b29.png" alt="activation f" style="zoom:80%;" /></p>


그럼 자세하게 들어가서 먼저 선형 변환(linear transformation)이 어떤 것인지 정의하고 가겠습니다.




## 선형 변환(linear transformation) 정의

$V$와 $W$가 어떤 $(1)$벡터 공간이고 둘 모두 $(2)$실수 집합 $(3)$상에 있다고 가정하겠습니다. 이때 함수 $(4) \mathcal{T}: V \rightarrow W$가 다음 두 조건을 만족할 때,

- 가산성(Additivity) : 모든 $x, y \in V$에 대해,  $\mathcal{T}(x+y) = \mathcal{T}(x)+ \mathcal{T}(y)$
- 동차성(Homogeneity) : 모든 $x \in V, c \in \Bbb{R}$에 대해, $\mathcal{T}(cx) = c\mathcal{T}(x)$

위 2가지 성질을 만족한다면, 함수 $\mathcal{T}$를 **선형 변환(linear transformation)** 이라고 부릅니다.


> (1) : 간단하게 말해서 벡터를 그릴 수 있는 공간입니다. 영상에서의 **좌표 평면**이라고 생각하시면 됩니다.
> 
> (2) : 정확히 표현하면 **같은 체(field)에 속해 있다**고 해야 하나, 이 글에선 실수만 다루기 때문에 실수 집합 상에 있다고 표현했습니다. 체의 예로는 실수 집합 $\Bbb{R}$, 유리수 집합 $\Bbb{Q}$, 복소수 집합 $\Bbb{C}$ 등이 있습니다.
> 
> (3) : **실수 집합 상에 있다**는 말은 $V$를 이루는 **원소들이 실수**라는 의미입니다. 예를 들어 실수 집합 상의 $V$가 어떤 벡터들의 집합이라고 했을 때, 그 벡터는 실수 벡터(벡터의 각 원소가 실수)가 됩니다.
> 
> (4): 정의역(domain)이 $V$ 이고 공역(codomain)이 $W$  인 함수 $\mathcal{T}$라는 의미입니다.



<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178720580-8fc58f96-918d-4df5-92ae-6cb68a0220f1.png" alt="activation f" style="zoom:80%;" /></p>



간단히 **'$\mathcal{T}$ 는 선형(linear) 이다.'** 라고 하기도 합니다. $\mathcal{T}$가 선형이라면 다음과 같은 성질을 가집니다.

> $\mathcal{T}(0)=0$
>
> 모든 $x,y \in V$ 와 $c \in \Bbb{R}$ 에 대해 다음 식과 동치입니다.
> $\mathcal{T}(cx+y)=c\mathcal{T}(x)+\mathcal{T}(y)$
> 
> 모든 $x,y \in V$ 에 대해 $\mathcal{T}(x-y)=\mathcal{T}(x)-\mathcal{T}(y)$ 는 $x_1, x_2, \dots, x_n \in V$ 과 $a_1, a_2, \dots, a_n \in \Bbb{R} $ 에 대해 다음의 식과 동치입니다.
> 
> $$ \mathcal{T} \biggl(\displaystyle\sum_{i=1}^n a_ix_i \biggl) = \displaystyle\sum_{i=1}^n a_i\mathcal{T}(x_i) $$ 

> 예를 하나 들어보죠. 다음과 같이 정의된 함수 $\mathcal{T} : R^2 \rightarrow R^2$ 는 선형일까요?
> 
> $\mathcal{T}(a_1,a_2) = (a_1+2a_2, a_2)$ 라고 정의를 하고, $c \in \Bbb{R}$이고 $(x_1, x_2), (y_1, y_2) \in \Bbb{R}^2$라고 하겠습니다. 그럼, $c(x_1,x_2)+(y_1,y_2)=(cx_1+y_1, cx_2+y_2)$ 이므로, 이를 이용해서 $\mathcal{T}(c(x_1,x_2)+(y_1,y_2))$ 를 구하면 다음과 같습니다.
> 
> $$ \begin{aligned} \mathcal{T}(c(x_1,x_2)+(y_1,y_2)) & = \mathcal{T}(cx_1+y_1, cx_2+y_2) \\ & = (cx_1+y_1 + 2(cx_2+y_2), cx_2+y_2) \end{aligned} $$
> 
> 또한,
> 
> $$ \begin{aligned} c\mathcal{T}(x_1,x_2)+\mathcal{T}(y_1,y_2) & = c(x_1+2x_2,x_2)+(y_1+2y_2,y_2)\\ & = (cx_1+2cx_2+y_1+2y_2, cx_2+y_2)\\ & = (cx_1+y_1 + 2(cx_2+y_2),cx_2+y_2) \end{aligned} $$
> 
> 이므로, $\mathcal{T}(c(x_1,x_2)+(y_1,y_2))=c\mathcal{T}(x_1,x_2)+\mathcal{T}(y_1,y_2)$ 입니다. 따라서 2번째 성질에 의해 $\mathcal{T}$ 는 **선형** 입니다.
>





## 비선형(Non-linear)

그렇다면 비선형은 뭘까요? 간단합니다. 선형이 아닌 함수를 **비선형(Non-linear) 함수**라고 합니다. 아래 함수 $f(x)$들을 살펴보고, 다음 질문에 답해 봅시다.

$1) f(x)=3x$

어떤 실수 $x,y,c$가 있다고 할 때, $f(cx+y)=3(cx+y)$이고, $cf(x)+f(y)=3cx+3y=3(cx+y)$이므로 $f$는 선형입니다.

<br>


$2) f(x)=x2$

어떤 실수 $x,y,c$가 있다고 할 때, $f(cx+y) = (cx+y)^2$이고, $cf(x)+f(y)=cx^2+y^2$이므로 $f$는 선형이 아닙니다.

<br>


$3) f(x)=\theta_0x_0 + \theta_1x_1(x=[x_0,  x_1]은 벡터)$

주어진 식을 벡터의 형태로 다음과 같이 표현할 수 있습니다.

$$
f(x)=θ_0x_0+θ_1x_1=[θ_0\ θ_1]⋅[x_0 \ x_1]=θx
$$

어떤 벡터 $x, y \in \Bbb{R}^2$와 어떤 실수 $c$가 있다고 할 때,

$$
\begin{aligned} f(cx+y)&=f([cx_0+y_0\ cx_1+y_1])\\ & =[θ_0\ θ_1]⋅[cx_0+y_0\ cx_1+y_1]\\&=θ_0(cx_0+y_0)+θ_1(cx_1+y_1) \end{aligned}
$$

이고,

$$
\begin{aligned} cf(x)+f(y)&=cθx+θy\\&=c(θ_0x_0+θ_1x_1)+θ_0y_0+θ_1y_1\\&=  θ_0(cx_0+y_0)+θ_1(cx_1+y_1) \end{aligned}
$$

이므로 $f$는 선형입니다.





<br>

# 비선형 함수를 쓰는 이유

그렇다면 왜 딥러닝에서는 비선형 활성화 함수를 주로 사용할까요? 한 문장으로 요약하자면, **"딥러닝 모델의 표현력을 향상시키기 위해서"** 입니다.

- 그럼 선형 활성화 함수를 사용하면 왜 표현력이 떨어지게 되는 걸까요?
- 레이어를 충분히 쌓는다면 선형 활성화 함수를 사용한 모델의 표현력을 향상시킬 수 있지 않을까요?

간단한 예시를 통해 알아가보도록 하겠습니다.


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178852881-9521a45b-ed04-4fa4-b4ea-80a6d6135b87.png" alt="activation f" style="zoom:100%;" /></p>


위 그림과 같이 퍼셉트론 3개로 구성된 모델이 있다고 가정하겠습니다. 입력값 $x$가 모델을 거치면 출력값 $y$가 됩니다. (여기서 입력값 $x$와 출력값 $y$는 스칼라값이고 $f$는 활성화 함수입니다.) 수식으로 표현하면 다음과 같습니다.

$$
y = f(w_3f(w_2f(w_1x)))
$$

여기서 $w_i$는 각 입력에 곱해지는 가중치이며, 편향값은 편의를 위해 $0$으로 두겠습니다. 이때 만약 $f$가 선형이라고 한다면 무슨 일이 일어날까요? $f$가 선형이기 때문에 선형 함수의 정의에 의해 $f(w_1x) = w_1f(x)$로 쓸 수 있기 때문에 이를 적용시키면 $w_1, w_2, w_3$을 아래의 식과 같이 합칠 수 있습니다.

$$
\begin{aligned} y = f(w_3f(w_2f(w_1x))) & = f(w_3f(f(w_1w_2x)))\\ & = f(f(f(w_1w_2w_3x)))\\ & = f(f(f(Wx)))\\ \end{aligned}
$$

여기서 $W = w_1w_2w_3$입니다.


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178853009-0055de4f-66b7-4b9f-9db1-170aadd0b07f.png" alt="activation f" style="zoom:100%;" /></p>


위 그림과 같이 $w_i$가 서로 자리를 바꿀 수 있는 것은 $w_i$가 스칼라이기 때문입니다. 편의를 위해 순서대로 나열했습니다 (즉, $w_1w_2 = w_2w_1$). 이것의 의미는 가중치의 업데이트가 $w_1, w_2, w_3$셋 전부에서 일어날 필요가 없다는 것입니다. 간단하게 예를 들어 보겠습니다.


$w_1, w_2, w_3$의 가중치를 모두 $1$로 초기화하고 모델을 훈련시켰을 때, 최종적으로 훈련된 모델의 가중치들이 $w_1', w_2', w_3'$라고 하겠습니다. 이것을 식으로 하면 다음과 같습니다.

$$
y = f(w_3'f(w_2'f(w_1'x)))
$$

함수 $f$가 선형인 것을 이용해 식을 다음과 같이 바꾸어 보겠습니다.

$$
y = f(f(f(W x)))
$$

여기서 $W = w_1' w_2' w_3'$입니다. 이 식은, 사실상, $w_1, w_2, w_3$의 가중치를 모두 $1$로 초기화하고 모델을 훈련시켰을 때, $w_2$와 $w_3$은 업데이트되지 않게 고정시키고  $w_1$만 업데이트한 것과 같습니다. 즉, $w_2$나 $w_3$의 가중치가 어떻게 변하는지와 상관없이  $w_1$만 잘 업데이트되면 결과는 같다는 것이죠. 그럼 나가아서 $f(f(f(W x)))$를 $f^\star(W x)$로 표현할 수도 있을까요? 이렇게 하기 위해선 그냥 $f$함수 3개를 하나의 합성함수로 만들어 주면 됩니다. 그런데 선형 함수들의 합성함수도 선형일까요?


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178853023-53f31411-9608-443e-af53-417f3d1754ea.png" alt="activation f" style="zoom:100%;" /></p>


네, 선형입니다. 그럼 어떻게 해서 선형이 되는지 증명해 보고 넘어가도록 할까요?





## 선형 변환의 합성함수에 관한 정리

---

$**V, W$그리고 $Z$가 실수 공간상의 벡터 공간이고, 함수 $\mathcal{T} : V \rightarrow W$와 함수 $\mathcal{U} : W \rightarrow Z$가 선형이라고 하면, 합성함수   $\mathcal{UT} : V \rightarrow Z$도 선형입니다**

**증명)**

$**x, y \in V$이고 $a \in \Bbb{R}$이라고 하겠습니다. 그럼,**

$$
\begin{aligned} \mathcal{UT}(ax+y) & = \mathcal{U}(\mathcal{T}(ax+y)) \\ & = \mathcal{U}(a\mathcal{T}(x)+\mathcal{T}(y)) \\ & = a\mathcal{U}(\mathcal{T}(x)) + \mathcal{U}(\mathcal{T}(y)) \\ & = a\mathcal{U}\mathcal{T}(x) + \mathcal{U}\mathcal{T}(y) \end{aligned}
$$

**이므로, 선형의 성질에 의해 $\mathcal{UT}$도 선형입니다.**

**우리는 이제 선형함수의 합성함수 또한 선형이라는 것을 알았습니다. 이 정리에 의해 우리는 이제 $f(f(f(W x)))$를 $f^\star(W x)$로 표현할 수 있습니다.**

**이것의 의미는 무엇일까요?**

**바로 3개의 노드를 1개로 줄여서 표현을 해도 결과가 달라지지 않는다는 것입니다.**






<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178853033-347f3752-a459-473d-96d9-1c31db1d8b7f.png" alt="activation f" style="zoom:80%;" /></p>

















