---
title:  "Pytorch 함수 및 팁 저장소(상시 추가)"
folder: "deep_learning"
categories:
  - deep_learning
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
---

### **Function**

**torch.roll(input, shifts, dims)**

roll함수는 input 매트릭스값을 원하는 dimension으로 shift하는 기능을 수행한다. 

``` python
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
print(x)

print(torch.roll(x, shifts=(3,1), dims=(0,1)))  # y축으로 3번 밀고 x축으로 1번민다는 의미
--------------------------------------------------
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
tensor([[3, 4],
        [5, 6],
        [7, 8],
        [1, 2]])
```
<br>

---

### **Tips**

#### **1\. torch.load\_state\_dict() 에서의  strict=False**

model\_b = torch.load\_state\_dict(model\_a.state\_dict(), strict=False) 를 사용하게 되면 model\_b의 레이어에 변화가 있더라도 기존의 model\_a와 같은 키 값을 가지는 weight(여기서 weight는 bias 포함한 의미) 들은 모두 model\_a 의 weight로 적용 할 수있다. (단 strict=False를 사용할때는 "같은 키 값"에 대해서만 적용이 되기 때문에 두 모델이 같은 레이어임에도 자신이 직접 모델을 만들면서 키값이 달라질 수 있기 때문에(함수, 클래스, 변수명 등에 의해 키값은 얼마든지 달라질 수 있음) 이때는 반목문을 통해 직접 하나씩 적용을 해주어야 한다.)

``` python
state_dict = model_b.state_dict()
param_names = list(state_dict.keys())

check_point = torch.load('model_a.pth.tar', map_location=config.DEVICE)
pretrained_state_dict = check_point['state_dict']
pretrained_param_names = list(check_point['state_dict'].keys())

# 키 값이 다르지만 model_a의 weight를 model_b에 모두 적용하고 싶을 때 (원하는 레이어만 적용할 때도 사용)
for i, param in enumerate(param_names):
    state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

model_b.load_state_dict(state_dict)

# 키 값이 같은 상황에서 model_b의 레이어의 변화가 있지만 기존의 레이어에는 적용하고 싶을 때
model_b.load_state_dict(check_point['state_dict'], strict=False)
```

#### **2\. torchvision의 transforms와 albumentation과의 혼란**

보통 torchvision의 transforms에서는 Normalize를 사용할 때 반드시 데이터를 min\_max scale(0~1)로 만들어주어야 하기 때문에 ToTensor를 먼저 사용한 후에 Normalize를 적용하는데 albumentation의 Normalize는 scaling과 normalize를 동시에 처리한다는 차이점이 있다.

albumentations에서는 ToTensor 대신 ToTensorV2를 사용하는데 ToTesorV2는 ToTensor와 마찬가지로 tensor형변환, channel dimension을 첫번째 차원으로 가져오는 역할을 하지만 min\_max scaling은 하지 않는다는 것을 명심하자.

(추가로 자료형 문제 때문에 albumentations에서 ToTensorV2를 Normalize 보다 앞에서 사용하면 에러가 발생한다.)

정리

-   torchvision transforms 에서는 Normalize 이전에 ToTensor를 사용할 것.
-   albumentations 에서는 Normalize 이후에 ToTensorV2를 사용할 것.