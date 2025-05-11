---
title:  "BentoML with Triton Inference Server"
folder: "tools"
categories:
  - tools
tags:
  - Backend
  - Segmentation
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
header:
  teaser: "/images/bentoml-result.png"
---

## Intro

안녕하세요 오랜만에 새로운 포스팅으로 찾아왔습니다. 이번 포스팅에서는 AI모델의 production 단계에서 많이 사용되는 python 기반의 서빙 프레임워크인 BentoML의 Integrations 중 하나를 소개해 드리려고 합니다.

그것은 바로 Triton Inference Server 입니다. 먼저 BentoML과 Triton Inference Server에 대해 간단하게 설명드리고 왜 BentoML에서 같은 서빙 프레임워크인 triton을 통합했는지 설명드리고 간단한 예제를 통해 사용법을 알아보겠습니다.

<br>

### What is BentoML?

BentoML은 AI 애플리케이션의 개발, 출시 및 확장을 용이하게 하도록 설계된 오픈 소스 플랫폼입니다. 이를 통해 개발팀은 Python을 사용하여 여러 모델과 모델 앞뒤에 필요한 전,후처리가 포함된 AI 애플리케이션을 신속하게 개발할 수 있습니다. 

또한 BentoML을 사용하면 사용량에 따라 애플리케이션을 효율적으로 쉽게 확장하여 다양한 수요를 처리할 수 있습니다. 자세한 내용은 [**공식문서**](https://docs.bentoml.org/en/v1.1.11/index.html){:target="_blank" style="color: red;" }를 참고해주세요.

### What is Triton Inference Server?

Triton Inference Server는 딥러닝 모델을 높은 성능으로 서빙을 할 수 있는 오픈소스 추론서버입니다. ONNX, TensorFlow, PyTorch, TensorRT 와 같은 다양한 딥러닝 프레임워크를 지원하며 다양한 모델 실행과 효율적인 배치 전략을 통해 하드웨어 활용도를 극대화할 수 있도록 최적화 설계되었습니다(C++ base).

Triton Inference Server는 복잡한 batching 처리 전략으로 사용 가능한 모든 리소스를 활용할 수 있는 고성능 추론 서버가 필요한 대규모 언어 모델을 제공하는 데 적합합니다.

### What Motivated The Triton Integration?

BentoML Python Runner의 단점 중 하나는 한 번에 하나의 스레드만 실행할 수 있는 CPython 인터프리터의 GIL(Global Interpreter Lock)입니다. GIL은 Python의 CPython 인터프리터에서 멀티 스레딩을 사용할 때 발생하는 주요 제약사항입니다.(멀티스레드가 동작 불가능한것은 아님.)

Python에서 GIL 문제가 발생하는 이유는 Python 인터프리터가 메모리 관리(garbage collection)와 관련된 동시성 문제를 방지하기 위해 설계된 방식 때문입니다. 여기에는 몇가지 중요한 포인트가 있습니다.

1. 메모리 관리와 안정성: Python은 내부적으로 메모리를 관리하는데, 이 과정에서 object reference count를 조작합니다. 여러 스레드가 동시에 object reference count를 변경할 경우 데이터 무결성에 문제가 생길 수 있습니다. GIL은 이러한 상황을 방지하기 위해 도입되었습니다. GIL이 활성화되면, 한 시점에 하나의 스레드만 Python 객체와 상호 작용할 수 있으므로, 메모리 관리가 안전하게 이루어 집니다.

2. 성능 저하: GIL의 존재는 하나의 프로세서에서 Python 프로그램의 병렬 실행을 제한합니다. 비록 I/O 바운드 작업(CPU 사용이 적고 대기시간이 김)에서는 GIL의 영향을 덜 받지만(파일을 읽거나 네트워크 요청을 기다리는 동안 일시적으로 GIL 해제하여 다른 스레드 실행가능), CPU 바운드 작업을 수행할 때는 여러 코어를 효율적으로 활용하지 못하게 되어 성능 저하를 초래할 수 있습니다. 이는 모델 추론과 같은 계산 집약적 작업에서 특히 문제가 될 수 있습니다.

3. Python Runner의 한계: python runner를 사용할 때, 모델 추론은 여전히 GPU나 멀티 스레드 CPU에서 병렬로 실행될 수 있지만, Python 코드(I/O, CPU bound logic)는 GIL로 인해 이러한 하드웨어의 병렬처리 능력을 완전히 활용할 수 없게 됩니다. 결과적으로, 하드웨어 자원이 충분히 활용되지 않아, 전반적인 성능과 처리량이 제한될 수 있습니다.

그렇기 때문에 <span style="color:teal">**Triton Inference Server 같은 C++ 기반의 추론서버를 사용함으로써, GIL의 영향을 받지 않는 병렬 처리와 고성능을 달성할 수 있습니다.**</span> Triton은 멀티 스레드와 멀티 프로세스 환경에서 최적화되어 있어, 모델 추론 작업을 위한 고성능 서빙을 제공할 수 있습니다. 이러한 이유로, BentoML 팀에서는 Python 기반의 서비스에서도 Triton과 같은 외부 추론 엔진을 통합하여 GIL의 제약을 극복하고 성능을 향상시키는 방법을 고안해 낸 것입니다.

BentoML v1.0.16부터 Triton Inference Server를 BentoML의 Runner 에서 사용할 수 있게 되었습니다.(2024년 2월 기준 latest:v1.1.11)

<br>

## Dive in

### Prerequisites

BentoML에서 Triton Inference Server를 활용하기 위해서는 먼저 Triton을 구동할 수 있는 환경이 필요합니다. 일반적으로 Triton은 Nvidia NGC에서 제공하는 도커 이미지를 활용하여 컨테이너 위에서 작동시킵니다.

```bash
docker pull nvcr.io/nvidia/tritonserver:<yy>.<mm>-py3
```

Triton은 매 달마다 버전(tag)이 업데이트 되고 있습니다. 그런데 만약 onnxruntime을 활용하여 추론을 하려는 분들을 tag를 22.12 이하로 사용해야 합니다. triton에서 23.01 버전부터는 CUDA 12.0 이상을 사용하기 때문입니다. 하지만 onnxruntime에서는 아직 CUDA 12.0을 지원하지 않아 호환성 문제가 있으니 주의해 주세요.

<br>

이미지를 정상적으로 가져왔다면 컨테이너에 접속해 봅시다. 

``` bash
docker run --gpus=all -it --shm-size=8G -p 3000:3000 \
-v <local model repository path>:<container model repository path> \
nvcr.io/nvidia/tritonserver:<yy>.<mm>-py3
```

일반적으로 triton만 사용할 때는 8000(http), 8001(grpc), 8002(metrics)를 포트포워딩 하지만 bentoml integrations를 통해 내부적으로 triton이 통합되기 때문에 bentoml의 API 기본 포트인 3000을 포트포워딩 해 줍니다.

triton의 model repository는 모델 파일과 모델의 로드 및 I/O 데이터 정의파일(.pbtxt)을 포함하는 파일 시스템 기반 pv(persistent volume)입니다. 때문에 triton을 사용하기 위해서는 사전에 model repository를 구성해야 합니다. (model repository 구성 방법에 대해 궁금하신 분들은 [**지난 포스팅**](https://visionhong.github.io/tools/YOLOv8-with-TensorRT-Nvidia-Triton-Server/){:target="_blank" style="color: red;" }을 참고해 주세요)

일반적으로 로컬 환경에서 model repository를 구성하고 위처럼 tritonserver 컨테이너에 볼륨 마운트를 해서 사용합니다.  

<br>

컨테이너 접속 후에 bentoml을 설치하겠습니다. bentoml에서 triton integrations를 사용하기 위해서 아래 명령어로 설치합니다.

``` bash
pip install -U "bentoml[triton]"
pip list | grep bentoml
```

설치된 bentoml의 버전이 1.0.16 이상인지 확인해주세요.


<br>

### Model Repository

BentoML에 들어가기전에 활용할 모델과 구조에 대해 간단하게 설명드리겠습니다. 제가 사용한 모델은 SAM(Segment Anything Model)의 경량화 모델인 Mobile-SAM 입니다. 

SAM 기반의 모델은 크게 Image Encoder와 Decoder로 구성됩니다. 저는 모델을 encoder와 decoder로 분리한 뒤에 onnx 모델로 변환하였습니다. 그리고 triton python backend를 통해 분리된 모델을 이어주고 전, 후처리 코드를 추가하였습니다. 그 결과 구조는 아래와 같습니다.

```
model_repository
├── mobile_sam
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
├── mobile_sam_encoder
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── mobile_sam_decoder
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

API를 통해 외부에서 encoder와 decoder에 요청을 보내도록 할 수 있지만 제가 의도한 방식은 python backend(model.py)에 정의된 workflow에서 encoder와 decoder가 필요한 시점에 각 모델에 추론요청을 보내는 것 입니다.

<br>

### Runner

BentoML에서 Runner는 원격 python worker에서 실행될 수 있고 독립적으로 확장될 수 있는 계산 단위를 나타냅니다.

Runner를 사용하면 `bentoml.Service`가 각각 자체 python worker에서 `bentoml.Runnable` 클래스의 여러 인스턴스를 병렬화할 수 있습니다. BentoServer가 시작되면 Runner worker process 그룹이 생성되고 `bentoml.Service` 코드에서 수행된 실행 메서드 호출이 해당 Runner worker 간에 예약됩니다.

bentoml에서 triton runner는 아래와 같이 생성할 수 있습니다. 

``` python
import bentoml

triton_runner = bentoml.triton.Runner(name="triton_runner",
                                      model_repository="/path/to/model_repository",
                                      tritonserver_type="http",
                                      cli_args=["--load-model=<model name>", "--model-control-mode=explicit"]
)

svc = bentoml.Service("triton-integration", runners=[triton_runner])
```

- model_repository : 도커 컨테이너를 실행할 때 볼륨 마운트했던 컨테이너 경로를 작성합니다. (S3와 같은 클라우드 저장소 활용 가능)
- tritonserver_type : 기본적으로 `bentoml.triton.Runner`는 gRPC 프로토콜을 사용하여 tritonserver를 실행합니다. HTTP/REST 프로토콜을 사용하려면 `tritonserver_type='http'`를 설정해야 합니다.
- cli_args : cli_args는 tritonserver 명령에 전달될 argument 목록입니다. 예를 들어 `--load-model` argument는 model repository에서 특정 모델을 로드하는 데 사용됩니다.
`--model-control-mode=explicit`은 `--load-model`에서 지정된 모델만 로드하며 나머지는 무시한다는 의미를 가집니다. 사용 가능한 argument에 대한 정보는 cli에서 `tritonserver --help`를 입력해서 확인할 수 있습니다.

<br>

### Service APIs (call runner with method)

API endpoint는 `@service.api` 와 같은 데코레이터 형태로 쉽게 생성할 수 있습니다. 이 데코레이터는 일반 Python 함수를 웹 API 엔드포인트로 변환합니다. 아래 예시를 통해 자세히 살펴보겠습니다.

``` python
from bentoml.io import JSON, Multipart, NumpyNdarray
import numpy as np
from tritonclient.http import InferInput, InferRequestedOutput
from utils.util import image2byte

@svc.api(
    input=JSON(), output= Multipart(mask=NumpyNdarray(),
                                    segmented_image=File(mime_type="image/png"),
                                    )
)
async def mobile_sam(input_data):
    image = check_condition(input_data, "input_image")

    pos_coords = np.array(input_data["pos_coords"]).astype(np.int64)
    neg_coords = np.zeros([1, 2], dtype=np.int64)
    labels = np.array(input_data["labels"]).astype(np.int64)
    image = image.astype(np.float32)
    
    inputs = [
        ("pos_coords", pos_coords, "INT64"),
        ("neg_coords", neg_coords, "INT64"),
        ("labels", labels, "INT64"),
        ("input_image", image, "FP32"),
    ]

    prepared_inputs = []
    for name, data, dtype in inputs:
        infer_input = InferInput(name, data.shape, dtype)
        infer_input.set_data_from_numpy(data, binary_data=True)  # Use binary_data=True for efficiency
        prepared_inputs.append(infer_input)

    outputs = [
        InferRequestedOutput("mask", binary_data=True),
        InferRequestedOutput("segmented_image", binary_data=True)
    ]

    InferResult = await triton_runner.infer(
        model_name="mobile_sam", model_version="1",
        inputs=prepared_inputs,
        outputs=outputs
    )

    return {
        "mask": InferResult.as_numpy("mask"),
        "segmented_image": image2byte(PIL.Image.fromarray(InferResult.as_numpy("segmented_image")))
    }
```

`@svc.api()` 안에는 데이터의 입출력 타입에 대한 정의가 필요합니다. 저는 Input을 JSON 형태로 받도록 했고 2개의 output을 처리하기 위해 `bentoml.io.Multipart` 를 사용하여 numpy array와 byte로 인코딩 된 png를 반환하도록 하였습니다.

api 아래 함수에 대해 간략하게 설명드리면 사용자의 요청으로 받은 JSON 으로부터 모델 입력 데이터 타입에 맞도록 각 데이터 변환하고 triton server에 요청을 보내는 방식입니다. (triton server에 추론을 요청하는 클라이언트 입장이라고 생각해주시면 됩니다.)

<br>

### Request & Response

서비스에 대한 정의가 완료되었다면 실제로 서빙을 하여 api를 통해 요청을 보낼 수 있게 됩니다.

``` bash
bentoml serve service.py:svc
```

cli에서 `bentoml serve` 명령어를 통해 서빙을 할 수 있습니다. 이때 `service.py:svc`는 파일명:서비스이름(데코레이터) 형식이어야 합니다.

서빙이 정상적으로 완료되었다면 `IP Adress:3000` 에 접속해서 아래와 같이 swagger를 확인할 수 있습니다.

![](/images/bentoml-swagger.png){: height="70%" width="70%"}

<br>

이제 API 요청을 보내기 위해 python에서 request body를 생성해 보겠습니다.

``` python
import json
import io
import requests
import base64
from requests_toolbelt.multipart import decoder
from PIL import Image

def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.getvalue()).decode()
    return base64_str

data = {}
data["pos_coords"] = [[400, 200]]
data["labels"] = [1]
data["input_image"] = encode_image(Image.open("sky.png"))

json_data = json.dumps(data)
```

위에서 API를 생성할 때 JSON Input을 받도록 했으므로 input 데이터를 모두 Json에 담아서 요청을 보내야 합니다. 저는 아래와 같이 3개의 input data를 json에 담았습니다.

- positive coordinates(x,y)
- Label(0:neg, 1:pos, 2: top-left, 3:bottom-right)
- base64 인코딩된 이미지

<br>

이제 request body를 생성해서 실제로 요청을 하고 응답을 확인해보겠습니다.

``` python
headers = {'Content-Type': 'application/json'}
url = 'http://<ip-address>:3000/mobile_sam'
response = requests.post(url, data=json_data, headers=headers)
multipart_data = decoder.MultipartDecoder.from_response(response)

mask = multipart_data.parts[0].content
segmented_image = Image.open(io.BytesIO(multipart_data.parts[1].content))
```

<div style="display: grid; grid-template-columns: repeat(2, 2fr); grid-gap: 10px;">

<div>
<img src="/images/bentoml-origin.png" alt="Image 1" style="max-width: 100%; height: auto;">
<p style="text-align: center;">input with coord</p>
</div>

<div>
<img src="/images/bentoml-result.png" alt="Image 2" style="max-width: 100%; height: auto;">
<p style="text-align: center;">segmented image</p>
</div>

</div>

정상적으로 추론이 완료되어 서버로부터 mask와 segmented image를 받았습니다. 이처럼 2개이상의 요청을 받을 때는 `requests_toolbelt.multipart.decoder` 를 사용하여 content를 분리해서 받을 수 있습니다.

<br>

## END

지금까지 BentoML의 Triton Inference Server integrations를 활용하는 방법에 대해 알아보았습니다. 이 integrations를 통해 BentoML의 기존 장점(빌드&배포, 플랫폼(YATAI))을 그대로 가져가면서 추론과정에서 GIL의 영향을 받지 않는 병렬 처리를 통해 조금 더 효율적으로 자원을 활용할 수 있습니다.

물론 이번 포스팅에서는 활용 방법에 대해서만 다루었지만 추후에 BentoML에서의 단순 서빙과 비교했을때 실질적으로 얼마나 더 효율적인지 실험을 통해 결과를 공유하는 시간을 가져보겠습니다.

Reference
- Docs: [https://docs.bentoml.org/en/v1.1.11/integrations/triton.html](https://docs.bentoml.org/en/v1.1.11/integrations/triton.html){:target="_blank" style="color: red;" }
- Post: [https://www.bentoml.com/blog/bentoml-or-triton-inference-server-choose-both](https://www.bentoml.com/blog/bentoml-or-triton-inference-server-choose-both){:target="_blank" style="color: red;" }


