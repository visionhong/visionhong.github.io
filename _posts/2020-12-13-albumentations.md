---
title:  "albumentations (Data Augmentation)"
folder: "deep_learning"
categories:
  - deep_learning
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
---

CNN 학습에서 데이터 augmentaion은 test 정확도를 향상시키는데에 중요한 역할을 한다.
현재 다양한 data augmentation 기법들이 존재하는데 빠르고 직관적이며 sequential하게 데이터 augmentation을 할 수 있도록 도와주는 라이브러리인 albumantations에 대해 글을 쓰려고 한다.
<br>



### Albumentations를 써야하는 이유 ?

\-  albumentations는 분류, semantic segmentation, instance segmentation, object detection 등 대부분의 컴퓨터 비전 에서의 작업을 지원한다.

\- 어떠한 이미지 데이터 유형이더라도 작업할 수 있는 통합 API를 제공한다.

\- 기존 이미지 데이터에서 새로운 학습 데이터를 생성하기 위해 70가지 이상의 다양한 augmentation기능을 갖추고 있다.

\- 그냥 빠르다.

\- Pytorch와 Tensorflow 등 인기있는 딥러닝 프레임워크에서 잘 동작한다. (albumentations는 Pytorch 소속이긴 하다.)

\- computer vision에 일가견이 있는 분들이 만들었다. (대부분 캐글 그랜드마스터,마스터)




### Simple Example

``` python
import albumentations as A
import matplotlib.pyplot as plt
import cv2


# augmentation 파이프라인 선언, p는 확률을 의미
transform = A.Compose([
    A.RandomCrop(width=200, height=100), #해당 크기로 잘라냄 /원래 이미지size = (340,148)
    A.HorizontalFlip(p=0.5), # 좌우반전
    A.RandomBrightnessContrast(p=1), # 밝기와 대비를 임의로 변경
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("고양이.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image = image)
transformed_image = transformed["image"]

plt.subplot(2,1,1)
plt.title("before")
plt.imshow(image)

plt.subplot(2,1,2)
plt.title("after")
plt.imshow(transformed_image)
plt.tight_layout()
```

![](/images/../images/2023-03-09-17-51-06.png){: .align-center}

<br>
코드실행 결과 위쪽 사진은 기존 이미지 이고 아래 사진은 albumentations 라이브러리를 이용해 augmentation한 이미지 이다.

이미지를 지정한 크기로 랜덤한곳을 자르고(RandomCrop), 좌우반전(HorizontalFlip)과 밝기및 대비를 랜덤으로 지정(RandomBrightnessContrast)하였다.

p는 해당 기법이 사용될 확률을 의미하며 코드에서 HorizontalFlip이 사용될 확률은 50% , RandomBrightnessContrast은 100%로 지정해주었다.(default 값은 0.5)

참고로 이미지는 (width x height x channel) shape에서 print를 할 수 있다. 좀 더 다양한 예제로 확인해보자  
<br>

``` python
import random

import cv2
import matplotlib.pyplot as plt
import albumentations as A

def visualize(image):
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(image)

image = cv2.imread("dog.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize(image)
```

![](/images/../images/2023-03-09-17-55-19.png){: .align-center}

<br>
``` python
transform = A.ShiftScaleRotate(p=0.5)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image)
```

![](/images/../images/2023-03-09-17-55-54.png){: .align-center}

<br>
여기서 3번째 줄에 \['image'\]를 해주는 이유는 transform을 거치면 딕셔너리를 반환하게되고 키값이 'image' 이기 때문이다.

``` python
transform(image=image)
```

\['image'\] 없이 위의 코드를 실행을하면 아래와 같은 결과가 나온다. 즉 \['image'\]는 이미지 array값만 가져오기 위해 쓴다는 의미

![](/images/../images/2023-03-09-17-56-13.png){: .align-center}
<br>
``` python
transform = A.Compose([
    A.CLAHE(p=1),  
    A.RandomRotate90(),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.75),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
])
random.seed(42)
augmented_image = transform(image=image)['image']
visualize(augmented_image)
```

![](/images/../images/2023-03-09-17-56-56.png){: .align-center}

<br>
``` python
transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

augmented_image = transform(image=image)['image']
visualize(augmented_image)
```

![](/images/../images/2023-03-09-17-57-11.png){: .align-center}
<br><br>
위의 코드에서 A.OneOf가 추가 되었는데 이 함수는 OneOf내에있는 기법들중에 랜덤으로 하나를 선택한다는 의미를 가진다. 5~8번째 코드를 보면 5번째줄에서 A.OneOf(\[ 으로 시작되고 8째줄에 \])로 닫히는데 이 말은 즉슨 6번째줄  A.IAAAdditiveGaussianNoise() 과 7번째줄A.GaussNoise() 이 둘중 하나의 기법만 사용을 하겠다는 뜻이다. 그리고 OneOf 내에 있는 p값은 해당 OneOf 자체가 실행 될 확률을 의미한다.

#### END

다음 포스팅에서는 데이터셋을 좀 더 쉽게 다룰 수 있도록 해주는 Pytorch의 Dataset과 DataLoader 를 다뤄보려고한다. Dataset을 정의하고 DataLoader에 이를 전달을 하면서 오늘배운 transform도 적용해볼 예정이다.
<br>

Reference : 
[https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)