---
layout: post
title: Pytorch Custom Dataset 사용하기
category: Pytorch
tag: Pytorch
---




메모리와 같은 하드웨어 성능의 한계 등의 이유로 한 번에 전체 데이터를 학습하는것은 힘들기 때문에 일반적으로 배치 형태의 묶음으로 데이터를 나누어 모델 학습에 이용됩니다. 또한 모델을 학습할 때 데이터의 특징과 사용 방법에 따라 학습 성능의 차이가 날 수 있으므로 데이터를 배치 형태로 만드는 법과 데이터를 전처리하는 방법에 대해서 알아보겠습니다.

<br>




# 1. 파이토치 제공 데이터 사용 : torchvision.datasets

```py
import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
```


<br>

- **torch** : 파이토치 기본 라이브러리 
- **torchvision** : 이미지와 관련된 파이토치 라이브러리
- **torchvision.transforms** : 이미지 전처리 기능들을 제공하는 라이브러리
- **from torch.utils.data import DataLoader, Dataset** : 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리

<br>

```py
# tr.Compose 내에 원하는 전처리를 차례대로 넣어준다.
transf = tr.Compose([
    tr.Resize(16),
    tr.ToTensor()
]) # 16x16으로 이미지 크기 변환 후 텐서 타입으로 변환

# torchvision.datasets에서 제공하는 CIFAR10 데이터를 불러온다.
# root : 다운로드 받을 경로를 입력
# train : Ture이면 학습 데이터를 불러오고 False이면 테스트 데이터를 불러옴
# transform : 미리 선언한 전처리를 사용
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transf)
```

<br>


**tr.Compose** 내에 **<u>원하는 전처리를 차례대로</u>** 넣어주면 됩니다. 예시에서는 16×16으로 이미지 크기 변환 후 텐서 타입으로 변환합니다. 만약 원본 이미지의 너비, 높이가 다를 경우 너비, 높이를 각각 지정을 해야하므로 **tr.Resize((16, 16))** 이라고 입력해야 합니다.

이후 **torchvision.datasets** 에서 제공하는 **CIFAR10** 데이터를 불러오고, 동시에 미리 선언한 전처리를 사용하기 위해 **transform=transf** 을 입력합니다.
```py
print(trainset[0][0].size())
```
```
[output]
torch.Size([3, 16, 16])
```


<br>


일반적으로 **데이터셋** 은 **<u>이미지와 라벨이 동시에 들어있는 튜플(이미지, 라벨) 형태</u>** 입니다.

- **trainset[0]** : 학습 데이터의 첫 번째 데이터로 이미지 한 장과 라벨 숫자 하나가 저장되어 있음.
- **trainset[0][0]** : 이미지
- **trainset[0][1]** : 라벨

<br>

현재 이미지 사이즈는 3×16×16 이며, 여기서 3은 채널 수를 말하고 16×16은 이미지의 너비와 높이를 의미합니다. 일반적인 컬러 사진은 RGB 이미지이기 때문에 채널이 3개이고 **(너비)x(높이)x(채널 수)** 로 크기가 표현되는 반면, **<span style="color:red">파이토치에서는 이미지 한 장이 (채널 수)X(너비)×(높이)로 표현</span>** 되니 유의해야합니다.
```py
# DataLoader는 데이터를 미니 배치 형태로 만들어 줍니다.
trainloader = DataLoader(trainset, batch_size=50, shuffle=True)
testloader = DataLoader(testset, batch_size=50, shuffle=False)
```


<br>



**DataLoader** 는 **<u>데이터를 미니 배치 형태로 만들어 줍니다.</u>** 따라서 배치 데이터에 관한 배치 사이즈 및 셔플 여부 등을 선택할 수 있습니다. 즉, **batch_size=50**, **shuffle=True** 은 무작위로 데이터를 섞어 한 번에 50개의 이미지를 묶은 배치로 제공하겠다는 의미입니다.
```py
# CIFAR10의 학습 이미지는 50,000장이고 배치 사이즈가 50장이므로 1,000은 배치의 개수가 됨
# 즉 trainloader가 잘 만들어졌다는 것을 단편적으로 알 수 있다.
len(trainloader)
```
```
[output]
1000
```


<br>

**CIFAR10** 의 학습 이미지는 50000장이고 배치 사이즈가 50장이므로 1000은 배치의 개수가 됩니다.
```py
# 일반적으로 학습 데이터는 4차원 형태로 모델에서 사용된다.
# (배치 크기)x(채널 수)x(너비)x(높이)
images, labels = next(iter(trainloader))
print(images.size())
```
```
[output]
torch.Size([50, 3, 16, 16])
```


<br>

배치 이미지를 간단히 확인하기 위해 파이썬에서 제공하는 **iter** 와 **next** 함수를 이용하면 됩니다. 이를 통해 **trainloader** 의 첫 번째 배치를 불러올 수 있습니다. **<span style="color:red">배치 사이즈는 (배치 크기)×(채널 수)×(너비)×(높이)를 의미</span>** 합니다. **즉, 배치 하나에 이미지 50개가 잘 들어가 있음을 알 수 있습니다.**
```py
oneshot = images[1].permute(1, 2, 0).numpy()
plt.figure(figsize=(2, 2))
plt.imshow(oneshot)
plt.axis("off")
plt.show()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/7d4359c8-482e-4f29-a5d4-78538629df11">
</p>


<br>


**image[1]** 의 크기는 **(3, 16, 16)** 입니다. **<u>이때 그림을 그려주기 위해서 채널 수가 가장 뒤로 가는 형태인 (16, 16, 3) 을 만들어야 하므로 permute() 함수를 이용하여 수정합니다.</u>** **permute(1,2,0)** 은 기존 차원의 위치인 0, 1, 2를 1, 2, 0으로 바꾸는 함수입니다. 따라서 0번째의 크기가 3인 텐서를 마지막으로 보니다. 마지막으로 **numpy()** 를 이용해 넘파이 배열로 변환합니다.

<br>





# 2. 같은 클래스 별로 폴더를 정리한 경우 : ImageFolder

**<u>데이터가 같은 클래스 별로 미리 폴더를 정리한 경우</u>**, **<span style="color:red">ImageFolder</span>** 하나로 **<u>개인 데이터를 사용할 수 있고, 또한 별도의 라벨링이 필요 없으며 폴더 별로 자동으로 라벨링을 합니다.</u>** 예를 들어 **class** 폴더에 **tiger**, **lion** 폴더(./class/tiger와 ./class/lion)를 미리 만들고나서 **ImageFolder** 에 상위 폴더 **./class** 를 입력하면 이미지와 라벨이 정리되어 데이터를 불러옵니다.
```py
# 데이터가 같은 클래스 별로 미리 폴더를 정리 된 경우, ImageFolder의 1줄 선언으로 개인 데이터를 사용할 수 있다.
# 별도의 라벨링이 필요 없으며 폴더 별로 자동으로 라벨링을 한다.
transf = tr.Compose([tr.Resize((128, 128)), tr.ToTensor()]) # 128x128 이미지 크기 변환 후 텐서로 만든다.
trainset = torchvision.datasets.ImageFolder(root='./class', transform=transf) # 커스텀 데이터 불러온다.
trainloader = DataLoader(trainset, batch_size=10, shuffle=False) # 데이터를 미니 배치 형태로 만들어 준다.
```
```py
images, labels = next(iter(trainloader))
print(images.size(), labels)
```
```
[output]
torch.Size([10, 3, 128, 128]) tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
```


<br>





# 3. 정리되지 않은 커스텀 데이터 불러오기

**ImageFolder** 를 이용하면 매우 간단하게 이미지 데이터를 사용할 수 있지만 여러 가지 이유로 사용이 불가한 경우가 있다.

- 라벨 별로 폴더 정리가 되어 있으면 매우 좋겠지만 그렇지 않은 경우가 많은 경우
- 정리를 하고 싶지만 다른 작업들과 공유된 데이터인 경우 폴더를 함부로 정리할 수 없는 경우
- 이미지 데이터라도 이미지가 아닌 텍스트, 리스트, 배열 등의 다른 형태로 저장되어 있는 경우

<br>



다음 양식은 커스텀 데이터를 불러오는 가장 기본적인 형태입니다.
```py
from torch.utils.data import Dataset

class 클래스명(Dataset):
    
    def __init__(self):
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

```


<br>

- **<span style="color:red">class 클래스명(Dataset):</span>** : **Dataset** 을 상속받아 **DataLoader** 에서 배치 단위로 불러올 수 있게 해줍니다.
- **<span style="color:red">def __init__(self):</span>** : 데이터 세팅에 필요한 것들을 미리 정의하는 역할을 합니다.
- **<span style="color:red">def __getitem__(self, index):</span>** : 이후 **DataLoader** 를 통해 샘플이 요청되면 인덱스에 해당하는 샘플을 찾아서 줍니다.
- **<span style="color:red">def __len__(self):</span>** : 크기를 반환합니다.

<br>



현재 32×32 크기인 RGB 컬러 이미지 100장과 그에 대한 라벨이 되어 있고 넘파이 배열로 정리가 되어 있다고 가정해보고 커스텀 데이터 세트 예시를 살펴보겠습니다.
```py
train_images = np.random.randint(256, size=(100, 32, 32, 3)) / 255 # (이미지 수)x(너비)x(높이)x(채널 수)
train_labels = np.random.randint(2, size=(100, 1)) # 라벨 수

# .....
# train_images, train_labels = preprocessing(train_images, train_labels)
# .....

print(train_images.shape, train_labels.shape)
```
```
[output]
(100, 32, 32, 3) (100, 1)
```


<br>


이미지 전처리 작업이 필요할 경우 openCV와 같은 라이브러리를 이용하여 이 곳에서 작업할 수도 있습니다. **preprocessing(train_images, train_labels)** 처럼 코드를 추가하여 전처리를 할 수 있는데, 이는 **torchvision.transforms** 라이브러리보다 OpenCV, SciPy와 같은 라이브러리가 더 많은 전처리 기술을 제공하며, 이미지를 미리 처리해 놓고 전처리 된 이미지를 살펴보면서 작업할 수 있습니다. 따라서 사용 목적과 편의성에 맞게 전처리를 어디서 할 지 정하면 됩니다.
```py
class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data) # 이미지 데이터를 FloatTensor로 변형
        self.x_data = self.x_data.permute(0, 3, 1, 2) # (이미지 수)x(너비)x(높이)x(채널 수) -> (배치 크기)x(채널 수)x(너비)x(높이)
        self.y_data = torch.LongTensor(y_data) # 라벨 데이터를 LongTensor로 변형
        self.len = self.y_data.shape[0] # 클래스 내의 들어 온 데이터 개수 

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] # 뽑아 낼 데이터를 적어준다.

    def __len__(self):
        return self.len # 클래스 내의 들어 온 데이터 개수

```


<br>

- **__init__**
    - **__init__** 에서 데이터를 받아 데이터를 텐서로 변환합니다.
    - 이때 원래 이미지의 크기가 **(100,32,32,3)** 이므로 **permute(0,3,1,2)** 함수를 통해 **(100,3,32,32)** 으로 바꿔줍니다. 파이토치에서는 **(배치 크기)x(채널 수)x(너비)x(높이)** 데이터가 사용되므로 원래 데이터 **(이미지 수)x(너비)x(높이)x(채널 수)** 를 변경해야만 합니다.
    - 입력 데이터의 개수에 대한 변수 **self.len** 을 만들어줍니다.
- **__getitem__**
    - 뽑아낼 데이터에 대해서 인덱스 처리를 하여 적어줍니다.
- **__len__**
    - 미리 선언한 **self.len** 를 반환할 수 있도록 넣어줍니다.

<br>

```py
train_data = TensorData(train_images, train_labels) # 텐서 데이터 불러오기 
train_loader = DataLoader(train_data, batch_size=10, shuffle=True) # 미니 배치 형태로 데이터 갖추기

images, labels = next(iter(train_loader))
print(images.size())
print(labels)
```
```
[output]
torch.Size([10, 3, 32, 32])
tensor([[1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1]])
```


<br>


이후 **TensorData** 클래스를 **train_data** 로 정의하여 **DataLoader** 에 넣어주면 배치 데이터의 형태로 사용할 수 있습니다.

<br>





# 4. 커스텀 데이터와 커스텀 전처리 사용하기

파이토치는 전처리 함수들을 제공하여 매우 편리하게 사용할 수 있습니다. 하지만 이미지의 경우 **PILImage** 타입이거나 **Tensor** 타입일 때만 사용이 가능하며, 또한 제공하지 않는 기능에 대해서는 직접 구현이 필요합니다. 이번 예시에서는 전처리 클래스 2개를 직접 정의하고 사용해보겠습니다.
```py
import torch
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# 32x32 컬러 이미지와 라벨이 각각 100장이 있다고 가정
train_images = np.random.randint(256, size=(100, 32, 32, 3)) / 255 # (이미지 수)x(너비)x(높이)x(채널 수)
train_labels = np.random.randint(2, size=(100, 1)) # 라벨 수
```
```py
# 1. 텐서 변환
class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs) # 텐서로 변환
        inputs = inputs.permute(2, 0, 1) # 크기 변환
        return inputs, torch.LongTensor(labels) # 텐서로 변환
```


<br>



텐서 변환 전처리 클래스를 정의합니다. 전처리는 **MyDataset** 클래스의 **sample** 을 불러와 작업하기 때문에 **__call__** 함수를 이용합니다. **ToTensor:** 는 입력 데이터를 텐서 데이터로 변환해 주고 학습에 맞는 크기로 변환하는 작업을 담당합니다. **torch.FloatTensor** 와 **torch.LongTensor** 를 이용해 텐서로 변환하고 **permute(2,0,1)** 을 이용해 크기를 변경하는데, 여기서 유의할 점은 **__call__** 함수는 입력값을 하나씩 불러오기 때문에 **permute(0, 3, 1, 2)** 이 아닌 **permute(2, 0, 1)** 로 코드를 작성해야합니다.


다음은 **CutOut** 전처리 클래스를 정의합니다. **CutOut** 은 이미지 내부에 무작위로 사각형 영역을 선택하여 0으로 만드는 데이터 증식 방법입니다.
```py
# 2. CutOut    
class CutOut:
    
    def __init__(self, ratio=.5):
        self.ratio = int(1/ratio)
           
    def __call__(self, sample):
        inputs, labels = sample
        active = int(np.random.randint(0, self.ratio, 1))
        
        if active == 0:
            _, w, h = inputs.size()
            min_len = min(w, h)
            box_size = int(min_len//4) # CutOut의 크기를 길이의 최솟값의 25%로 설정한다.
            idx = int(np.random.randint(0, min_len-box_size, 1)) # idx를 통해 CutOut 박스의 좌측 상단 꼭지점 위치를 정해준다.
            inputs[:, idx:idx+box_size, idx:idx+box_size] = 0 # 해당 정사각형 영역의 값을 0으로 대체한다.
        
        return inputs, labels
```


<br>


**ToTensor** 와 다르게 외부에서 **CutOut** 발생 비율을 받기 위해 **__init__** 함수를 사용하여 **ratio** 를 받습니다. 기본 **ratio** 는 0.5로 세팅하면 불러온 이미지에 대해서 50% 확률로 **CutOut** 를 발현합니다.

**__call__** 함수에서는 샘플을 받습니다. **active** 는 정수를 뽑으며, 50%일 경우 0과 1 중 하나를 뽑게 되고 0이면 **CutOut** 를 발현하고 0이 아니면 원본을 그대로 내보내게 됩니다. **CutOut** 이 발현될때 **inputs.size()** 를 통해 이미지의 너비와 높이를 받아 최솟값을 구하고, CutOut의 크기를 길이의 최솟값의 25%로 설정한 후, **CutOut** 박스의 좌측 상단 꼭지점 위치를 정하여 해당 정사각형 영역의 값을 0으로 대체합니다.


**MyDataset** 에서 전처리를 추가해보겠습니다.
```py
# 3.3에서 사용한 양식을 그대로 사용하되 전처리 작업을 할 수 있도록 transform을 추가한다. 
class MyDataset(Dataset):
    
    def __init__(self, x_data, y_data, transform=None):
        
        self.x_data = x_data # 넘파이 배열이 들어온다.
        self.y_data = y_data # 넘파이 배열이 들어온다.
        self.transform = transform
        self.len = len(y_data)
        self.tensor = ToTensor()
    
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample) # self.transform이 None이 아니라면 전처리를 작업한다.
        else:
            sample = self.tensor(sample)
        
        return sample
    
    def __len__(self):
        return self.len
```


<br>



**__init__** 의 입력값에 **transform=None** 을 추가하는데, **transform=None** 는 아무 것도 적지 않으면 전처리를 사용하지 않겠다는 의미입니다. 만약 **transform** 이 **None** 이 아니라면 **__getitem__** 에서 **sample** 을 반환하기 전에 전처리를 할 수 있도록 if문을 작성하고, **transform=None** 일 경우에는 텐서 변환은 기본적으로 하도록 구성합니다.
```py
trans = tr.Compose([ToTensor(), CutOut()]) 
dataset1 = MyDataset(train_images,train_labels, transform=trans)
train_loader1 = DataLoader(dataset1, batch_size=10, shuffle=True)

images1, labels1 = next(iter(train_loader1))
print(images1.size()) # 배치 및 이미지 크기 확인
```
```
[output]
torch.Size([10, 3, 32, 32])
```

<br>


**ToTensor()** 와 **tr.ToTensor()** 의 차이를 살펴보면, 앞서 사용한 **tr.ToTensor()** 는 **torchvision.transforms** 를 이용한 파이토치 메소드를 이용한 것이고, **ToTensor()** 는 위에서 정의된 메소드를 사용한 것입니다. **CutOut** 은 괄호에 아무 값도 없으므로 발현 비율의 기본값인 0.5로 **CutOut** 이 시행됩니다. 그리고 정의된 전처리를 입력한 데이터 세트를 만들고 **DataLoader** 를 사용합니다.
```py
import torchvision

def imshow(img):
    plt.figure(figsize=(10, 100))
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()

imshow(torchvision.utils.make_grid(images1, nrow=10))
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d0da58ed-7452-4f62-b346-5b234f5accdd">
</p>

<br>



그리드를 만들어주는 **torchvision.utils.make_grid** 를 사용하기 위해 **torchvision** 을 불러온 후 그림을 그리기 위해 **(채널 수, 너비, 높이)** 인 이미지 크기를 **permute(1, 2, 0)** 으로 **(너비, 높이, 채널 수)** 로 변경하고 **numpy()** 를 이용하여 넘파이 배열로 변환합니다. 첫번째 이미지를 확대해서 살펴보면 다음과 같습니다.
```py
imshow(images1[0])
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c1637168-9624-492f-9a8d-f411692737fb">
</p>

<br>






# 5. 커스텀 데이터와 파이토치 제공 전처리 사용하기


텐서 변환과 같은 전처리는 파이토치에서 제공하는 전처리를 사용하면 편리합니다. 하지만 앞서 언급했듯이 파이토치의 **torchvision.transforms** 에서 제공되는 많은 전처리는 **PILImage** 타입 또는 **텐서** 일 경우 사용할 수 있습니다. 따라서 기능은 있는데 데이터 타입이 다른 경우는 **PILImage** 타입으로 변환하여 제공된 전처리를 사용할 수 있습니다.
```py
# torchvision.transforms은 입력 이미지가 일반적으로 PILImage 타입이나 텐서일 경우에 동작한다.
# 현재 데이터는 넘파이 배열이므로, 텐서 변환 후 tr.ToPILImage()을 이용하여 PILImage 타입으로 만들어 준다.

class MyTransform:
    
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2, 0, 1)
        labels = torch.FloatTensor(labels)

        transf = tr.Compose([
            tr.ToPILImage(),
            tr.Resize(128),
            tr.ToTensor()
        ])
        final_output = transf(inputs)      
        
        return final_output, labels  
```


<br>


전처리 클래스 **MyTransform** 을 정의하여 원하는 전처리를 모두 작성합니다. **tr.Compose** 는 차례대로 전처리 작업을 하므로 가장 첫 번째에 **tr.ToPILImage()** 를 넣어 이미지 타입을 바꿔줄 수 있습니다. 이후 불러온 샘플을 전처리 작업에 넣어줍니다.
```py
dataset2 = MyDataset(train_images, train_labels, transform=MyTransform())
train_loader2 = DataLoader(dataset2, batch_size=10, shuffle=True)

images2, labels2 = next(iter(train_loader2))
print(images2.size()) # 배치 및 이미지 크기 확인
```
```
[output]
torch.Size([10, 3, 128, 128])
```


<br>

**MyDataset** 의 전처리에 **MyTransform()** 을 넣어주면 전처리가 완료됩니다.
```py
imshow(torchvision.utils.make_grid(images2, nrow=10))
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/801bbbdc-1510-4b1c-ada6-5ada950c6f2f">
</p>

<br>






# 6. 커스텀 전처리와 파이토치에서 제공하는 전처리 함께 사용하기

위에서 사용한 **CutOut** 과 달리 다음 **CutOut** 은 라벨은 받지 않고 이미지를 받아 처리하도록 세팅합니다. 그 이유는 **Compose** 내부에 있는 제공된 전처리는 이미지만 받아서 처리하기 때문에 그 양식을 맞춰 주어야 하기 때문입니다. 이후 **MyDataset** 과 **MyTransform** 을 정의하겠습니다. 우리가 만든 CutOut은 텐서나 넘파이 배열 타입 모두 작동을 하게 만들었지만 PILImage 타입에서는 타입 오류가 나므로 **tr.ToTensor()** 뒤에 **CutOut** 을 배치합니다. 
```py
class CutOut:
    
    def __init__(self, ratio=.5):
        self.ratio = int(1/ratio)
           
    def __call__(self, inputs):

        active = int(np.random.randint(0, self.ratio, 1))
        
        if active == 0:
            _, w, h = inputs.size()
            min_len = min(w, h)
            box_size = int(min_len//4)
            idx = int(np.random.randint(0, min_len-box_size, 1))
            inputs[:, idx:idx+box_size, idx:idx+box_size] = 0

        return inputs
```
```py
class MyDataset(Dataset):
    
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data # 넘파이 배열이 들어온다.
        self.y_data = y_data # 넘파이 배열이 들어온다.
        self.transform = transform
        self.len = len(y_data)
        self.tensor = ToTensor()
    
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample) # self.transform이 None이 아니라면 전처리를 작업한다.
        else:
            sample = self.tensor(sample)
        
        return sample
    
    def __len__(self):
        return self.len       
```
```py
class MyTransform:
    
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2, 0, 1)
        labels = torch.FloatTensor(labels)

        transf = tr.Compose([
            tr.ToPILImage(),
            tr.Resize(128),
            tr.ToTensor(),
            CutOut()
        ])
        final_output = transf(inputs)
        
        return final_output, labels
```

<br>



이제 전처리를 적용한 결과를 확인하겠습니다.
```py
import torch
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# 32x32 컬러 이미지와 라벨이 각각 100장이 있다고 가정
train_images = np.random.randint(256, size=(100, 32, 32, 3)) / 255 # (이미지 수)x(너비)x(높이)x(채널 수)
train_labels = np.random.randint(2, size=(100, 1)) # 라벨 수

dataset3 = MyDataset(train_images, train_labels, transform=MyTransform())
train_loader3 = DataLoader(dataset3, batch_size=10, shuffle=True)

images3, labels3 = next(iter(train_loader3))
print(images3.size()) # 배치 및 이미지 크기 확인
```
```
[output]
torch.Size([10, 3, 128, 128])
```


<br>

```py
imshow(torchvision.utils.make_grid(images3, nrow=10))
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/9e2cf0a9-fcc0-40ba-ba01-fc3afc7b9c32">
</p>

<br>



전처리를 적용하지않은 결과를 확인하겠습니다.
```py
dataset3 = MyDataset(train_images, train_labels)
train_loader3 = DataLoader(dataset3, batch_size=10, shuffle=True)

images3, labels3 = next(iter(train_loader3))
print(images3.size()) # 배치 및 이미지 크기 확인
```
```
[output]
torch.Size([10, 3, 32, 32])
```


<br>

```py
imshow(torchvision.utils.make_grid(images3, nrow=10))
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/faa3379c-48ab-4ebd-a2ac-82ed2dd330bf">
</p>

<br>







다음 결과를 통해 CIFAR10 데이터가 배치 10개씩 나눠지고 이미지 사이즈를 128로 늘린 뒤 텐서로 변환되고 50% 확률로 무작위 선택하여 CutOut을 적용한 것을 알 수 있습니다.
```py
transf = tr.Compose([
    tr.Resize(128),
    tr.ToTensor(),
    CutOut()
])
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transf
)
trainloader = DataLoader(
    trainset,
    batch_size=10,
    shuffle=True
)
```
```py
images, labels = next(iter(trainloader))
print(images.size()) # 배치 및 이미지 크기 확인
```
```
[output]
torch.Size([10, 3, 128, 128])
```

<br>

```py
imshow(torchvision.utils.make_grid(images, nrow=10))
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/4f728ead-fa93-44d1-b182-63db7309964d">
</p>





