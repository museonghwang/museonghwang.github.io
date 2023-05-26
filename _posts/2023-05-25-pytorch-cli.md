---
layout: post
title: pytorch project workflow 구성하기
category: Pytorch
tag: Pytorch
---





주피터 노트북을 활용하면 실행 결과를 바로 확인할 수 있기 때문에 데이터 분석과 같이 각 셀의 결과에 따라 해야 하는 일이 바뀌는 경우에 적합합니다. 하지만 머신러닝 프로젝트는 해야 할 작업이 명확하고 반복되므로 **py** 확장자를 가진 파이썬 스크립트로 제작하여 **CLI(command line interface)** 환경에서 작업을 수행하는 것이 좀 더 바람직합니다.



따라서 데이터 분석 과정을 제외한 머신러닝 프로젝트 대부분의 과정은 CLI 환경에서 수행됩니다. **<span style="color:red">특히 모델링 및 하이퍼파라미터 튜닝 작업 시에는 반복적인 실험이 수행되기 때문에 코드를 수정하여 실험을 수행하는 것이 아니라 CLI 환경에서 파이썬 스크립트 실행과 함께 실행 파라미터를 넣어주어 실험을 수행하도록 하는 것이 더 낫습니다.</span>** 그러므로 주피터 노트북을 활용한 실습을 벗어나 실무 환경에서 머신러닝 프로젝트를 수행하는 것처럼 프로젝트 또는 솔루션을 설계하고 구현할 수 있어야 합니다.

<br>




# 머신러닝 프로젝트 파일 구조 예시

가장 간단한 형태의 머신러닝 프로젝트를 구현하면 다음과 같은 구조를 지닐 것입니다. 역할에 따른 파일 이름은 예시로 든 것입니다.

| 파일명 | 설명 |
| ---- | --------------------------------------------|
| **model.py** | 모델 클래스가 정의된 코드 |
| **trainer.py** | 데이터를 받아와 모델 객체를 학습하기 위한 trainer가 정의된 코드 |
| **dataloader.py** | 데이터 파일을 읽어와 전처리를 수행하고 신경망에 넣기 좋은 형태로 변환하는 코드 |
| **train.py** | 사용자로부터 하이퍼파라미터를 입력받아 필요한 객체들을 준비하여 학습을 진행 |
| **predict.py** | 사용자로부터 학습된 모델과 추론을 위한 샘플을 입력받아 추론을 수행 |

<br>




기능에 따라 각 모듈을 나누어 클래스를 정의하고 다른 프로젝트에 재활용하기도 하며, 또한 모델 개선이나 기타 수정 작업이 필요할 때 코드 전체를 바꾸거나 할 필요 없이 필요한 최소한의 부분만 수정하여 사용할 수 있습니다. 실제로 이에 따라 하나의 템플릿을 구성해 놓으면 계속해서 일부만 수정해서 사용할 수 있습니다. 다음 그림은 앞에서 소개한 파일들이 어떤 식으로 상호작용하는지 나타낸 것입니다.



<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/5209a6a9-fc1e-4736-a8a3-f18f83e85cf6">
</p>

<br>





**train.py** 는 사용자가 학습을 진행할 때 직접 실행할 파이썬 스크립트 파일로, 해당 파일을 실행하면 사용자로부터 필요한 하이퍼파라미터를 입력받아 각 클래스의 객체를 생성하고 학습을 진행합니다. 사용자는 이 **<span style="color:red">train.py 를 통해서 코드 수정 없이 다양한 하이퍼파라미터들을 변경해가며 반복적인 실험을 수행</span>** 할 수 있습니다.



예를 들어 **CLI** 환경에서 다음과 같이 **train.py** 를 실행하고 하이퍼파라미터를 **argument** 를 통해 전달합니다. 아래의 실행 명령은 모델 가중치 파일이 저장될 파일 경로와 모델의 깊이 그리고 드롭아웃의 확률을 **train.py** 에 넘겨줍니다. 그러면 **<span style="color:red">프로그램은 이러한 하이퍼파라미터를 넘겨받아 코드 수정 없이 다양한 실험을 수행하도록 구현</span>** 되어야 할 것입니다.


```sh
$ python train.py --model_fn ./models/model.pth --n_layers 10 --dropout 0.3
```



<br>


또한 **trainer** 는 **data loader** 로 부터 준비된 데이터를 넘겨받아 모델에 넣어 학습과 검증을 진행하는 역할을 수행합니다. 이렇게 학습이 완료되면 모델의 가중치 파라미터는 보통 **피클(pickle)** 형태로 다른 필요한 정보(e.g. 모델을 생성하기 위한 각종 설정 및 하이퍼파라미터)들과 함께 파일로 저장됩니다.



그러면 **predict.py** 는 저장된 피클 파일을 읽어와서 모델 객체를 생성하고 학습된 가중치 파라미터를 그대로 복원합니다. 그리고 사용자로부터 추론을 위한 샘플이 주어지면 모델에 통과시켜 추론 결과를 반환합니다.



이처럼 실제 머신러닝 프로젝트는 반복적으로 수행되는 작업을 효율적으로 수행하기 위해서 복잡한 구조를 잘게 쪼개어 각각 모듈들로 구현하도록 합니다. 복잡한 머신러닝 프로젝트일지라도 결국 데이터와 모델을 불러와서 학습하고 기학습된 모델을 가지고 추론을 수행한다는 역할은 근본적으로 같습니다.

<br>




# 머신러닝 프로젝트 Workflow

실제 머신러닝 프로젝트를 수행하듯 각 기능별 모듈들을 구성하여 MNIST 분류를 구현하겠습니다.

1. **문제 정의**
    - 단계를 나누고 simplity
    - x와 y를 정의
2. **데이터 수집**
    - 문제 정의에 따른 수집
    - 필요에 따라 레이블링
3. **데이터 전처리 및 분석**
    - 형태를 가공
    - 필요에 따라 EDA 수행
4. **알고리즘 적용**
    - 가설을 세우고 구현/적용
5. **평가**
    - 실험 설계
    - 테스트셋 구성
6. **배포**
    - RESTful API를 통한 배포
    - 상황에 따라 유지/보수



<br>



## 1. 문제 정의

손글씨 순자를 인식하는 함수 $f^*$ 를 근사계산하고 싶습니다. 따라서 근사계산한 모델 함수 $f_{\theta}$ 는 이미지를 입력받아 숫자 레이블을 출력하도록 구성될 것입니다. 이 모델을 만들기 위해서 우리는 손글씨 숫자를 수집하고 이에 대한 레이블링도 수행, 즉 MNIST 데이터셋을 구축합니다.



<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/6cbd5e4f-e6bf-4a3b-986c-890cca4cef1f">
</p>

<br>



위 그림과 같이 한 장의 이미지는 28×28 개의 흑백(grayscale) 픽셀로 구성되어 있습니다. 따라서 우리가 만들 함수의 입력은 784차원의 벡터가 되고, 출력은 각 숫자 클래스별 확률값이 되도록 구현될 것입니다.

<br>



## 2. 데이터 수집

MNIST라는 공개 데이터셋을 활용하므로 매우 수월하지만, 실무 환경에서는 데이터가 없거나 데이터가 있더라도 레이블이 존재하지 않는 상황도 맞이하게 될 것입니다. 데이터 수집 및 레이블링 작업을 수행하기 위한 두 가지 선택지가 있는데, 첫 번째로 직접 데이터 수집 및 레이블링을 진행하는 것이 있으며, 두 번째로는 외주를 맡기거나 단기 계약직 등을 고용하는 방법도 있습니다. 둘 중 어떤 선택을 하든지 업무의 크기를 산정해야 하고 예산을 준비하는 작업도 필요합니다.

<br>



## 3. 데이터 전처리

이제 데이터셋을 학습용과 검증용 그리고 테스트용으로 나누는 작업을 수행할 차례입니다. MNIST 데이터셋의 경우 기본적으로 60,000장의 **학습 데이터셋(training dataset)** 과 10,000장의 **테스트 데이터셋(test dataset)** 으로 구분되어 있습니다. 따라서 테스트셋은 주어진 10,000장을 사용하도록 하고 60,000장을 8:2의 비율로 학습 데이터셋과 **검증 데이터셋(validation dataset)** 으로 나누어 줍니다. 그러면 최종적으로 다음 그림과 같이 학습 데이터셋 48,000장, 검증 데이터셋 12,000장 그리고 테스트 데이터셋 10,000장을 얻을 수 있습니다.


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/eda1cf93-1d99-4ff4-ada6-de960577124c">
</p>

<br>




데이터를 분할한 이후 데이터 전처리를 수행합니다. 이때 데이터의 성격에 따라 필요한 전처리가 매우 다르므로, 따라서 본격적으로 전처리를 수행하기에 앞서 데이터가 어떤 분포와 형태를 띠고 있는지 면밀히 분석해야 합니다. 다양한 전처리들은 데이터의 종류와 형태 그리고 상태에 따라서 다르게 적용되며 크게 다음 그림과 같이 나뉠 수 있습니다.


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/bf182e10-cf7b-40a5-bb05-4a20d62b3144">
</p>

<br>




일부 전처리 기법들은 데이터를 기반으로 파라미터가 결정되는데, 데이터 기반의 전처리 기법은 학습 데이터셋 기준으로 수행되어야 합니다. 즉, 학습 데이터만을 가지고 평균과 표준편차를 계산한 뒤 학습/검증/테스트 데이터셋에 일괄 적용하는 형태가 되어야 합니다. 만약 전체 데이터셋을 기반으로 평균과 표준편차를 계산하고 정규화 스케일을 적용하게 되면 테스트셋을 보고 테스트를 평가하는 것과 다를 바 없습니다. **결론적으로 전처리는 학습/검증/테스트 데이터셋 분할 작업 이후에 수행하는 것이 바람직합니다.**

다행히도 MNIST 데이터셋의 경우 별다른 전처리가 필요하지 않습니다. 0에서 255사이의 값으로 채워진 픽셀 값을 255로나누어 0에서 1사이의 값으로 정규화해주는 작업 정도면 충분합니다.

<br>



## 4. 알고리즘 적용

데이터 전처리 과정에서 수행된 분석을 통해 데이터의 분포나 성질을 파악할 수 있었을 것입니다. 따라서 우리는 분석 결과를 바탕으로 알맞은 가설을 설정하고 알고리즘 구현 및 적용해야 합니다. 이 과정에서 분석 결과에 따라 가장 적절한 머신러닝 알고리즘을 적용하면 됩니다.


신경망 내부의 자세한 구조 결정에 앞서 회귀 문제인지 분류 문제인지에 따라 손실 함수와 마지막 계층의 활성 함수가 결정됩니다. 또한 계층의 개수, 활성 함수의 종류, 정규화 방법 등의 하이퍼파라미터가 남아 있는데 이들을 결정하기 위한 프로세스는 다음과 같습니다.

1. **신경망 외형 구성**
    - 오버피팅이 발생할 때까지 계층을 쌓는다.
2. **활성 함수 결정**
3. **Regularization 결정**
4. **Optimizer 결정**
5. **평가(Evaluation)**
    - 평가를 통해 베이스라인(baseline)을 구축
6. **튜닝(Tuning)**
    - 점진적으로 성능을 개선


<br>

먼저 적당한 선택으로 초기 하이퍼파라미터를 설정한 다음에 오버피팅이 발생할 때까지 신경망을 깊고 넓게 만듭니다. 오버피팅이 발생하는 것을 확인함으로써 데이터셋의 복잡한 데이터를 신경망이 충분히 학습할 만한 수용 능력을 지녔음을 알 수 있습니다. 또한 오버피팅이 발생하더라도 매 에포크마다 검증 데이터셋에 대한 손실 값을 추적하고 있으므로 큰 문제가 되지않습니다. 이후에 적절한 score metric을 적용하여 모델을 평가하고 모델의 성능을 수치화합니다.


여기까지가 한 번의 모델링 과정을 거친 것이 되고 이후 하이퍼파라미터를 수정하며, 이 과정을 반복하여 모델의 성능을 점진적으로 개선합니다. 또는 단순한 하이퍼파라미터 수정만으로는 충분한 성능 개선이 이루어지지 않는다면 성능 저하 문제의 원인에 대한 적절한 가설을 설정하고 모델의 구조를 바꾸는 등 수정을 거쳐 성능을 개선할 수도 있습니다.

<br>



# 5. 평가

서비스 또는 배포를 위해서, 그리고 모델의 성능 개선을 위해서 공정하고 객관적인 평가가 수행되어야합니다.


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/81270b7f-871b-4da0-8300-5b78e81a9e53">
</p>

<br>



평가 결과에 따라 언더피팅이 의심될 경우에는 모델의 수용 능력을 더 키우는 방향으로 하이퍼파라미터를 튜닝하고 오버피팅으로인해 일반화 성능이 저하되는 것이 우려될 때에는 정규화 기법을 강화하는 방향으로 튜닝하면서 학습과 평가를 반복 수행하게 될 것입니다.

| 범주 | 학습 데이터셋 | 검증 데이터셋 | 테스트 데이터셋 |
| --- | ---------- | --------- | ----------- |
| 가중치 파라미터 | 결정 | 검증 | 검증 |
| 하이퍼파라미터 | - | 결정 | 검증 |
| 알고리즘 | - | - | 결정 |


<br>


이와 같이 모델 성능 개선 작업이 종료되고 나면 테스트 데이터셋을 활용하여 평가를 수행함으로써 진정한 모델(또는 알고리즘)의 성능을 공정하게 평가할 수 있습니다.

<br>



## 6. 배포

이제 알고리즘이 실전에 투입될 준비가 되었다고 판단되면 본격적으로 배포 과정에 들어가게 됩니다.


<br>
<br>




# MNIST 학습 구조 설계


## 1. 모델 구조 설계

MNIST 분류기를 만들 것이기 때문에 모델은 28×28 크기의 이미지를 펼쳐진 784차원의 벡터로 입력받아 각 숫자 클래스별 확률 값을 반환해야 합니다. 다음은 구현할 모델의 구조를 그림으로 나타낸 것입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/55eb1f6c-224d-4cb6-822b-07eb7004b70e">
</p>

<br>



반복되는 계층을 쌓아 구현할 것이기 때문에 반복되는 부분을 깔끔하게 블록(block) 클래스로 정의하고 쌓아 올릴 것입니다. 하나의 블록 내에는 선형 계층과 비선형 활성 함수인 리키렐루 그리고 정규화를 위한 배치정규화 계층이 차례대로 들어가 있습니다. 이후에는 분류를 위해 클래스 개수(MNIST의 경우에는 10차원)만큼의 차원으로 변환하는 선형 계층과 로그소프트맥스 함수를 배치할 것입니다.


이렇게 하면 모델은 각 클래스 (MNIST의 경우에는 0부터 9까지의 숫자)별 로그 확률 값을 뱉어낼 것이고 이것을 정답 원 핫 벡터와 비교하면 손실 값을 계산할 수 있습니다. 이때 그냥 소프트맥스 함수가 아닌 로그소프트맥스 함수를 활용했기 때문에 손실 값 계산을 위해서 NLL 손실 함수를 사용해야 합니다.


흥미로운 점은 모델을 구현할 때 우리가 풀고자 하는 MNIST와 연관된 하드코딩을 거의 하지 않을 것이므로 이 모델이 MNIST 이외의 분류 문제에도 바로 적용 가능할 것이라는 것입니다. 따라서 만약 다른 문제에 바로 코드를 적용하고자 한다면 데이터 로더 부분만 수정하면 거의 그대로 동작할 것입니다. **<span style="color:red">이와 같이 애초에 구현 단계에서 최소한의 수정으로 최대한 재활용과 확장이 가능하도록 설계하고 구현하는 것이 매우 중요하며, 또한 이를 위해서 각 기능의 모듈이 잘 나뉘어 독립적으로 구현되어 있어야 합니다.</span>**

<br>



## 2. 학습 과정

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0fdd2c3e-735a-4760-a048-e6d266a8f965">
</p>

<br>



오른쪽 그림과 같이 학습/검증/테스트 데이터셋으로 구성된 데이터를 **Trainer** 모듈에 넣어주어 모델을 학습할 것입니다. 이때 학습을 위한 전체 **n_epochs** 만큼의 반복이 진행될 것이고, 이것은 왼쪽 그림의 맨 바깥쪽 싸이클로 표현되어 있습니다.


하나의 에포크는 학습을 위한 부분과 검증을 위한 부분으로 나누어져 있을 것입니다. 학습과 검증은 각각 미니배치별로 이터레이션을 위한 반복문으로 구현되어 있을 텐데 이중 학습 과정에서는 손실 계산 후 역전파와 경사하강법을 통한 가중치 파라미터 업데이트 과정이 포함되어 있을 것입니다.


<br>



## 3. 파일 구조

| 파일명 | 설명 |
| ----- | --- |
| **model.py** | nn.Module을 상속받아 모델 클래스를 정의 |
| **predict.ipynb** | 학습이 완료된 모델 피클 파일을 불러와 샘플을 입력받아 추론 수행 |
| **train.py** | 사용자가 학습을 진행하기 위한 진입 지점 |
| **trainer.py** | 모델 객체와 데이터를 받아 실제 학습 이터레이션을 수행하는 클래스를 정의 |
| **utils.py** | 프로그램 내에서 공통적으로 활용되는 모듈을 모아 놓은 스크립트 |


<br>


현재 굉장히 작은 프로젝트이므로 이 파일들 전부를 한 디렉터리 내에 위치하게 할 것이지만 나중에 프로젝트 규모가 커지고 파일이 많아진다면 디렉터리 구조를 추가하여 좀 더 효율적으로 관리해야 합니다.

<br>


---


<br>



# 분류기 모델 구현하기


분류기(classifier) 모델 클래스를 정의하도록 하겠습니다. 반복되는 형태를 블록으로 만들어 크기만 다르게 한 후에 필요한 만큼 쌓을 수 있도록 구현할 것입니다. 먼저 블록을 서브 모듈(sub-module)로 넣기 위해 클래스로 정의합니다.
```py
class Block(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y
```



<br>




하나의 블록은 **nn.Linear** 계층, **nn.LeakyReLU** 활성 함수, **nn.BatchNorm1d** 계층 또는 **nn.Dropout** 계층 이렇게 3개로 이루어져 **nn.Sequential** 에 차례대로 선언되어있는 것을 볼 수 있습니다.


눈여겨보아야 할 점은 **get_regularizer** 함수를 통해 **use_batch_norm** 이 **True** 이면 **nn.BatchNorm1d** 계층을 넣어주고, **False** 이면 **nn.Dropout** 계층을 넣어준다는 것입니다. 이렇게 선언된 **nn.Sequential** 은 **self.block** 에 지정되어 **forward** 함수에서 피드포워드가 되도록 간단히 구현됩니다.


모델은 이렇게 선언된 블록을 반복해서 재활용할 수 있습니다. 다음 코드는 최종 모델로써 앞에서 선언된 블록을 재활용하여 아키텍처를 구성하도록 되어 있습니다. 참고로 이 모델은 이후에 작성할 코드에서 MNIST 데이터를 28×28 이 아닌 784차원의 벡터로 잘 변환했을 거라고 가정했습니다. 따라서 추후에 잊지 말고 올바른 데이터를 넣어주도록 구현해주어야 합니다.
```py
class ImageClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):
        
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
```


<br>


마찬가지로 **nn.Sequential** 을 활용하여 블록을 필요한 만큼 쌓도록 합니다. 여기에서 클래스 선언 시에 입력받은 **hidden_sizes** 를 통해 필요한 블록의 개수와 각 블록의 입출력 크기를 알 수 있습니다. 따라서 **hidden_sizes** 를 활용하여 **for** 반복문 안에서 **Block** 클래스를 선언하여 **blocks** 라는 리스트에 넣어줍니다. 이렇게 채워진 **blocks** 를 **nn.Sequential** 에 바로 넣어주고 이어서 각 클래스별 로그 확률 값을 표현하기 위한 **nn.Linear** 와 **nn.LogSoftmax** 를 넣어줍니다. 이후 **self.layers** 에 선언한 **nn.Sequential** 객체를 넣어주어 **forward** 함수에서 피드포워드하도록 구현하였음을 확인할 수 있습니다.

<br>




# 데이터 로딩 구현하기

파이토치에는 MNIST를 쉽게 로딩할 수 있도록 코드를 제공하고 있습니다. 따라서 MNIST 파일을 직접 손으로 다운로드해서 코드상에 경로를 지정하여 읽어오는 일 따위는 하지 않아도 됩니다. 다음 함수는 MNIST를 로딩하는 함수입니다.
```py
def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data',
        train=is_train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y
```


<br>



x와 y에는 이미지 데이터와 이에 따른 클래스 레이블이 담겨있을 것입니다. 다만 x의 경우 원래 28×28 이므로 **flatten** 이 **True** 일 때 **view** 함수를 통해 784차원의 벡터로 바꿔주는 것을 볼 수 있습니다. 또한, 원래 각 픽셀은 0에서 255까지의 그레이 스케일 데이터이기 때문에 이를 255로 나누어서 0에서 1사이의 데이터로 바꿔줍니다.


MNIST는 본래 60,000장의 학습 데이터와 10,000장의 테스트 데이터로 나누어져 있습니다. 따라서 60,000장의 학습 데이터를 다시 학습 데이터와 검증데이터로 나누는 작업을 수행해야 합니다. 다음의 함수는 해당 작업을 수행합니다.
```py
def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y
```


<br>




# Trainer 클래스 구현하기


앞에서 작성한 모델 클래스의 객체를 학습하기 위한 트레이너 클래스를 살펴볼 차례입니다. 클래스의 각 메서드(함수)를 살펴보도록 하겠습니다. 다음은 클래스의 가장 바깥에서 실행될 train 함수입니다.
```py
def train(self, train_data, valid_data, config):
    lowest_loss = np.inf
    best_model = None

    for epoch_index in range(config.n_epochs):
        train_loss = self._train(train_data[0], train_data[1], config)
        valid_loss = self._validate(valid_data[0], valid_data[1], config)

        # You must use deep copy to take a snapshot of current best weights.
        if valid_loss <= lowest_loss:
            lowest_loss = valid_loss
            best_model = deepcopy(self.model.state_dict())

        print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
            epoch_index + 1,
            config.n_epochs,
            train_loss,
            valid_loss,
            lowest_loss,
        ))

    # Restore to best model.
    self.model.load_state_dict(best_model)
```


<br>



앞에서 코드를 살펴보기 전 그림을 통해 전체 과정을 설명을 했었습니다. 그때 학습과 검증 등을 아우르는 큰 루프(loop)가 있었고 학습과 검증 내의 작은 루프가 있었습니다. **train** 함수 내의 **for** 반복문은 큰 루프를 구현한 것입니다. 따라서 내부에는 **self._train** 함수와 **self._validate** 함수를 호출하는 것을 볼 수 있습니다.


그리고 곧이어 검증 손실 값에 따라 현재까지의 모델을 따로 저장하는 과정도 구현되어있습니다. 현재까지의 최고 성능 모델을 **best_model** 변수에 저장하기 위해서 **state_dict** 라는 함수를 사용하는 것을 볼 수 있는데, 이 **state_dict** 함수는 모델의 가중치 파라미터 값을 **json** 형태로 변환하여 리턴합니다. 이 **json** 값의 메모리를 **best_model** 에 저장하는 것이 아니라 값 자체를 새로 복사하여 **best_model** 에 할당하는 것을 볼 수있습니다.


그리고 학습이 종료되면 **best_model** 에 저장된 가중치 파라미터 **json** 값을 **load_state_dict** 를 통해 **self.model** 에 다시 로딩합니다. 이 마지막 라인을 통해서 학습 종료후 오버피팅이 되지 않은 가장 좋은 상태의 모델로 복원할 수 있게 됩니다.


이번에는 **_train** 함수를 살펴봅니다. 이 함수는 한 이터레이션의 학습을 위한 **for** 반복문을 구현했습니다.
```py
def _train(self, x, y, config):
    self.model.train()

    x, y = self._batchify(x, y, config.batch_size)
    total_loss = 0

    for i, (x_i, y_i) in enumerate(zip(x, y)):
        y_hat_i = self.model(x_i)
        loss_i = self.crit(y_hat_i, y_i.squeeze())

        # Initialize the gradients of the model.
        self.optimizer.zero_grad()
        loss_i.backward()

        self.optimizer.step()

        if config.verbose >= 2:
            print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

        # Don't forget to detach to prevent memory leak.
        total_loss += float(loss_i)

    return total_loss / len(x)
```


<br>


함수의 시작 부분에서 잊지 않고 **train()** 함수를 호출하여 모델을 학습 모드로 전환하는 것을 확인할 수 있습니다. 만약 이 라인이 생략된다면 이전 에포크의 검증 과정에서 추론 모드였던 모델 그대로 학습에 활용될 것입니다. **for** 반복문은 작은 루프를 담당하고 해당 반복문의 내부는 미니배치의 피드포워드와 역전파 그리고 경사하강법에의한 파라미터 업데이트가 담겨있습니다.


마지막으로 **config.verbose** 에 따라 현재 학습 현황을 출력합니다. **config** 는 가장 바깥의 **train.py** 에서 사용자의 실행 시 파라미터 입력에 따른 설정값이 들어있는 객체입니다.


**train** 함수의 가장 첫 부분에 **_batchify** 함수를 호출하는 것을 볼 수 있습니다. 다음 **_batchify** 함수는 매 에포크마다 SGD를 수행하기 위해 셔플링 후 미니배치를 만드는 과정입니다.
```py
def _batchify(self, x, y, batch_size, random_split=True):
    if random_split:
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)

    x = x.split(batch_size, dim=0)
    y = y.split(batch_size, dim=0)

    return x, y
```


<br>


검증 과정에서는 **random_split** 이 필요 없으므로 **False** 로 넘어올 수 있음을 유의해야합니다. 다음 코드는 검증 과정을 위한 **_validate** 함수입니다.
```py
def _validate(self, x, y, config):
    # Turn evaluation mode on.
    self.model.eval()

    # Turn on the no_grad mode to make more efficintly.
    with torch.no_grad():
        x, y = self._batchify(x, y, config.batch_size, random_split=False)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            if config.verbose >= 2:
                print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            total_loss += float(loss_i)

        return total_loss / len(x)
```


<br>


대부분 **train** 과 비슷하게 구현되어 있음을 알 수 있습니다. 다만 가장 바깥쪽에 **torch.no_grad()** 가 호출되어 있는 것에 대해 유의해야합니다.


<br>





# train.py 구현하기

**train.py** 를 통해 다양한 파라미터를 시도하고 모델을 학습할 수 있습니다. **CLI** 환경에서 바로 **train.py** 를 호출할 것이며 그러고 나면 **train.py** 의 다음 코드가 실행될 것입니다.
```py
if __name__ == '__main__':
    config = define_argparser()
    main(config)
```


<br>


먼저 **define_argparser()** 라는 함수를 통해 사용자가 입력한 파라미터들을 **config** 라는 객체에 저장합니다. 다음 코드는 **define_argparser** 함수를 정의한 코드입니다.
```py
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config
```


<br>



**argparse** 라이브러리를 통해 다양한 입력 파라미터들을 손쉽게 정의하고 처리할 수 있습니다. **train.py** 와 함께 주어질 수 있는 입력들은 다음과 같습니다.

| 파라미터 이름 | 설명 | 기본 설정값 |
| ---------- | --- | -------- |
| model_fn | 모델 가중치가 저장될 파일 경로 | 없음. 사용자 입력 필수 |
| gpu_id | 학습이 수행될 그래픽카드 인덱스 번호 (0부터 시작) | 0 또는 그래픽 부재 시 -1 |
| train_ratio | 학습 데이터 내에서 검증 데이터가 차지할 비율 | 0.8 |
| batch_size | 미니배치 크기 | 256 |
| n_epochs | 에포크 개수 | 20 |
| n_layers | 모델의 계층 개수 | 5 |
| use_dropout | 드롭아웃 사용 여부 | False |
| dropout_p | 드롭아웃 사용 시 드롭 확률 | 0.3 |
| verbose | 학습 시 로그 출력의 정도 | 1 |


<br>



**model_fn** 파라미터는 **required=True** 가 되어 있으므로 실행 시 필수적으로 입력되어야 합니다. 이외에는 디폴트 값이 정해져 있어서 사용자가 따로 지정해주지 않으면 디폴트 값이 적용됩니다. 만약 다른 알고리즘의 도입으로 이외에도 추가적인 하이퍼파라미터의 설정이 필요하다면 **add_argument** 함수를 통해 프로그램이 입력받도록 설정할 수 있습니다. 이렇게 입력받은 파라미터들은 다음과 같이 접근할 수 있습니다.
```py
config.model_fn
```


<br>



앞서 모델 클래스를 정의할 때 **hidden_sizes** 라는 리스트를 통해 쌓을 블록들의 크기를 지정할 수 있었습니다. 사용자가 블록 크기들을 일일이 지정하는 것은 어쩌면 번거로운 일이 될 수 있기 때문에, 사용자가 모델의 계층 개수만 정해주면 자동으로 등차수열을 적용하여 **hidden_sizes** 를 구해봅시다. 다음의 **get_hidden_sizes** 함수는 해당 작업을 수행합니다.
```py
def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes
```


<br>


이제 학습에 필요한 대부분이 구현되었습니다. 이것들을 모아서 학습이 진행되도록 코드를 구현하면 됩니다. 다음의 코드는 앞서 구현한 코드를 모아서 실제 학습을 진행 과정을 수행하도록 구현한 코드입니다.
```py
def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)
```


<br>



MNIST에 특화된 입출력 크기를 갖는 것이 아닌 벡터 형태의 어떤 데이터도 입력받아 분류할 수 있도록 **input_size** 와 **output_size** 변수를 계산하는 것에 주목하세요. **MNIST** 에 특화된 하드코딩을 제거하였기 때문에 **load_mnist** 함수가 아닌 다른 로딩 함수로 바꿔치기하면 이 코드는 얼마든지 바로 동작할 수 있습니다.


사용자로부터 입력받은 설정(configuration)을 활용하여 모델을 선언한 이후에 아담 옵티마이저와 NLL 손실 함수도 함께 준비합니다. 그리고 트레이너를 초기화한 후 **train** 함수를 호출하여 불러온 데이터를 넣어주어 학습을 시작합니다. 학습이 종료된 이후에는 **torch.save** 함수를 활용하여 모델 가중치를 **config.model_fn** 경로에 저장합니다.

<br>



## 코드 실행

파이썬 스크립트로 작성되었기 때문에 **CLI** 환경에서 실행할 수 있습니다. **train.py** 에서 **argparse** 라이브러리를 활용하여 사용자의 입력을 파싱하여 인식할 수 있습니다. 다만, 처음 사용하거나 오랜만에 실행하는 경우 어떤 입력 파라미터들이 가능한지 기억이 나지 않을 수도 있습니다. 그때에는 입력 파라미터 없이 다음과 같이 실행하거나 **'--help'** 파라미터를 넣어 실행하면 입력 가능한 파라미터들을 확인할 수 있습니다.
```bash
$ python train.py
usage: train.py [-h] --model_fn MODEL_FN [--gpu_id GPU_ID]
                [--train_ratio TRAIN_RATIO] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--n_layers N_LAYERS] [--use_dropout]
                [--dropout_p DROPOUT_P] [--verbose VERBOSE]
train.py: error: the following arguments are required: --model_fn
```


<br>



입력할 파라미터를 정해서 다음과 같이 직접 실행하면 정상적으로 학습이 진행되는 것을 볼 수 있습니다.
```bash
$ python train.py --model_fn tmp.pth --gpu_id -1 --batch_size 256 --n_epochs 20 --n_layers 5
```
```
[output]
Train: torch.Size([48000, 784]) torch.Size([48000])
Valid: torch.Size([12000, 784]) torch.Size([12000])
ImageClassifier(
  (layers): Sequential(
    (0): Block(
      (block): Sequential(
        (0): Linear(in_features=784, out_features=630, bias=True)
        (1): LeakyReLU(negative_slope=0.01)
        (2): BatchNorm1d(630, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Block(
      (block): Sequential(
        (0): Linear(in_features=630, out_features=476, bias=True)
        (1): LeakyReLU(negative_slope=0.01)
        (2): BatchNorm1d(476, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): Block(
      (block): Sequential(
        (0): Linear(in_features=476, out_features=322, bias=True)
        (1): LeakyReLU(negative_slope=0.01)
        (2): BatchNorm1d(322, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): Block(
      (block): Sequential(
        (0): Linear(in_features=322, out_features=168, bias=True)
        (1): LeakyReLU(negative_slope=0.01)
        (2): BatchNorm1d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): Linear(in_features=168, out_features=10, bias=True)
    (5): LogSoftmax(dim=-1)
  )
)
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: False
    lr: 0.001
    maximize: False
    weight_decay: 0
)
NLLLoss()
Epoch(1/20): train_loss=1.9642e-01  valid_loss=1.0231e-01  lowest_loss=1.0231e-01
Epoch(2/20): train_loss=8.1590e-02  valid_loss=9.6950e-02  lowest_loss=9.6950e-02
Epoch(3/20): train_loss=5.8084e-02  valid_loss=8.0420e-02  lowest_loss=8.0420e-02
Epoch(4/20): train_loss=4.1818e-02  valid_loss=8.2402e-02  lowest_loss=8.0420e-02
Epoch(5/20): train_loss=3.3787e-02  valid_loss=7.7139e-02  lowest_loss=7.7139e-02
Epoch(6/20): train_loss=2.6198e-02  valid_loss=8.4243e-02  lowest_loss=7.7139e-02
Epoch(7/20): train_loss=2.4709e-02  valid_loss=7.6167e-02  lowest_loss=7.6167e-02
Epoch(8/20): train_loss=2.5783e-02  valid_loss=9.7144e-02  lowest_loss=7.6167e-02
Epoch(9/20): train_loss=2.1204e-02  valid_loss=8.3943e-02  lowest_loss=7.6167e-02
Epoch(10/20): train_loss=1.4652e-02  valid_loss=7.6630e-02  lowest_loss=7.6167e-02
Epoch(11/20): train_loss=1.7424e-02  valid_loss=8.1460e-02  lowest_loss=7.6167e-02
Epoch(12/20): train_loss=1.4826e-02  valid_loss=8.0137e-02  lowest_loss=7.6167e-02
Epoch(13/20): train_loss=1.2403e-02  valid_loss=9.0082e-02  lowest_loss=7.6167e-02
Epoch(14/20): train_loss=1.2167e-02  valid_loss=9.2580e-02  lowest_loss=7.6167e-02
Epoch(15/20): train_loss=1.4440e-02  valid_loss=9.5186e-02  lowest_loss=7.6167e-02
Epoch(16/20): train_loss=9.6789e-03  valid_loss=7.9955e-02  lowest_loss=7.6167e-02
Epoch(17/20): train_loss=9.0307e-03  valid_loss=7.3424e-02  lowest_loss=7.3424e-02
Epoch(18/20): train_loss=9.2683e-03  valid_loss=8.4777e-02  lowest_loss=7.3424e-02
Epoch(19/20): train_loss=9.8202e-03  valid_loss=8.8804e-02  lowest_loss=7.3424e-02
Epoch(20/20): train_loss=8.7659e-03  valid_loss=9.8113e-02  lowest_loss=7.3424e-02
```


<br>


**CLI** 환경에서 사용자가 **train.py** 를 실행할 때 간단한 입력으로 필요한 파라미터를 주도록 하면 간편하기도 할뿐더러 실수를 줄일 수 있을 것입니다. **MNIST** 분류기의 성능을 높이는 것도 중요하지만 이처럼 **MNIST** 분류기를 좀 더 연구하기 위해 편리한 실험 환경을 갖추는 것도 매우 중요합니다.


<br>




# predict.ipynb 구현하기


학습을 마치면 가중치 파라미터가 담긴 파일이 **torch.save** 함수를 활용하여 피클 형태로 저장되어 있을 것입니다. 그럼 이제 해당 모델 파일을 불러와서 추론 및 평가를 수행하는 코드를 구현해야 합니다. 보통은 **train.py** 처럼 **predict.py** 를 만들어서 일반 파이썬 스크립트로 짤 수도 있지만 좀 더 손쉬운 시각화를 위해 주피터 노트북을 활용하도록 하겠습니다. 만약 단순히 추론만 필요한 상황이라면 **predict.py** 를 만들어 추론 함수를 구현한 후에 **API** 서버 등에서 랩핑(wrapping) 하는 형태로 구현할 수 있을 것입니다. 다음은 **torch.load** 를 활용하여 **torch.save** 로 저장된 파일을 불러오기 위한 코드입니다.
```py
model_fn = "./tmp.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load(fn, device):
    d = torch.load(fn, map_location=device)
    
    return d['model'], d['config']
```


<br>




**map_location** 을 통해서 내가 원하는 디바이스로 객체를 로딩하는 것에 주목하세요. 만약 **map_location** 을 쓰지 않는다면 자동으로 앞서 학습에 활용된 디바이스로 로딩될 것입니다. 같은 컴퓨터라면 크게 상관없지만 만약 다른 컴퓨터일 때 GPU가 없거나 개수가 다르다면 문제가 생길 수 있습니다. 예를 들어 GPU 4개짜리 컴퓨터에서 3번 GPU를 활용해서 학습된 파일인데 추론 컴퓨터에는 0번 GPU까지만 있는 상황이라면 문제가 발생할 것입니다.


다음은 추론을 직접 수행하는 코드를 **test** 함수로 구현한 모습입니다. **eval()** 함수를 활용하여 잊지 않고 모델을 추론 모드로 바꿔주었습니다. 또한 **torch.no_grad()** 를 활용하여 효율적인 텐서 연산을 위한 부분도 확인할 수 있습니다.
```py
def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)

        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))
```

```py
def test(model, x, y, to_be_shown=True):
    model.eval()
    
    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))
        
        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)
        
        if to_be_shown:
            plot(x, y_hat)
```


<br>



다만 현재 이 코드의 문제점은 미니배치 단위로 추론을 수행하지 않는다는 것입니다. **MNIST** 와 같이 작은 데이터에 대해서는 크게 문제 되지 않을 수도 있지만 만약 테스트셋이 한 번에 연산하기에 너무 크다면 **OOM(Out of Memory)** 에러가 발생할 것입니다. 이 부분은 **for** 반복문을 통해 간단하게 구현할 수 있습니다. 다음 코드는 앞서 선언한 코드를 불러와서 실제 추론을 수행하는 코드입니다.
```py
model_dict, train_config = load(model_fn, device)

# Load MNIST test set.
x, y = load_mnist(is_train=False)
x, y = x.to(device), y.to(device)

input_size = int(x.shape[-1])
output_size = int(max(y)) + 1

model = ImageClassifier(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=get_hidden_sizes(input_size,
                                  output_size,
                                  train_config.n_layers),
    use_batch_norm=not train_config.use_dropout,
    dropout_p=train_config.dropout_p,
).to(device)

model.load_state_dict(model_dict)

test(model, x, y, to_be_shown=False)
```
```
[output]
Accuracy: 0.9837
```


<br>



**load_state_dict** 는 **json** 형태의 모델 가중치가 저장된 객체를 실제 모델 객체에 로딩하는 함수입니다. 앞에서 트레이너 코드를 설명할 때에도 사용된 것을 볼 수 있었습니다. 무엇보다 **load_state_dict** 를 사용하기에 전에 **ImageClassifer** 객체를 먼저 선언하여 **model** 변수에 할당하는 것을 볼 수 있습니다. 즉, 이렇게 생성된 **model** 객체는 임의로 초기화된 가중치 파라미터 값을 가지고 있을 텐데, 이것을 **load_state_dict** 함수를 통해 학습이 완료된 기존의 가중치 파라미터 값으로 바꿔치기하는것으로 이해할 수 있습니다.


마지막에 **test** 함수에 전체 테스트셋을 넣어주어 전체 테스트셋에 대한 테스트 성능을 확인할 수 있습니다. 10,000장의 테스트셋 이미지에 대해서 98.37%의 정확도로 분류를 수행하는 것을 볼 수 있습니다.


아직 모델을 거의 튜닝하지 않은 것이기 때문에 검증 데이터셋을 활용하여 하이퍼파라미터 튜닝을 수행한다면 미미하게나마 성능 개선을 할 수도 있을 것입니다. 중요한 점은 절대로 테스트셋을 기준으로 하이퍼파라미터 튜닝을 수행해선 안된다는 것입니다. 다음은 실제 시각화를 위해서 일부 샘플에 대해 추론 및 시각화를 수행하는 코드와 그 결과를 보여주고 있습니다.
```py
n_test = 2
test(model, x[:n_test], y[:n_test], to_be_shown=True)
```
```
[output]
Accuracy: 1.0000
Predict: 7.0
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/47257ae7-da29-4429-b367-34f30bd5ce83">
</p>

<br>



샘플에 대해서 정확도 100%가 나오고 시각화된 결과를 눈으로 확인해보았을 때에도 정답을 잘 맞히는 것을 확인할 수 있습니다. 단순히 테스트셋에 대해서 추론 및 정확도 계산만 하고 넘어가기 보다 이처럼 실제 샘플을 뜯어보고 눈으로 확인하면서 틀린 것들에 대한 분석을 해야 합니다.



<br>





# 마무리


단순히 주피터 노트북을 활용해서 한 셀씩 코드를 적어나가는 것이 아니라 문제를 해결하기 위한 최적의 알고리즘과 하이퍼파라미터를 연구하고 찾을 수 있는 환경 구축하는 방법을 살펴보았습니다. 이와 같이 프로젝트 환경을 구축하게 되면 추후 다른 프로젝트를 수행할 때에도 최소한의 수정을 거쳐 재활용할 수 있게 됩니다. 정리하면 현재 우리가 구현한 프로젝트는 다음과 같은 요구사항을 반영하고 있습니다.

- 효율적으로 실험을 반복해서 수행할 수 있어야 한다.
- 모델 아키텍처가 바뀌어도 바로 동작할 수 있어야 한다.
- 하이퍼파라미터를 바꿔서 다양한 실험을 돌릴 수 있어야 한다.
- 코드의 일부분이 수정되어도 다른 부분은 큰 수정이 없도록 독립적으로 동작해야 한다.


<br>


---



<br>


# 코드 정리


## model.py : 모델 클래스 정의

```py
import torch
import torch.nn as nn


class Block(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y

    
class ImageClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):
        
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
```


<br>



## utils.py : 프로그램 내에서 공통적으로 활용되는 모듈을 모아 놓은 스크립트

```py
import torch


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data',
        train=is_train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y


def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes
```


<br>



## trainer.py : 모델 객체와 데이터를 받아 실제 학습 이터레이션을 수행하는 클래스를 정의

```py
from copy import deepcopy
import numpy as np

import torch

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _batchify(self, x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)
```


<br>



## train.py : 사용자가 학습을 진행하기 위한 진입 지점

```py
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)

```


<br>





