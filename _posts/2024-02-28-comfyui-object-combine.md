---
title:  "(ComfyUI) How to append object in image?"
folder: "image_generation"
categories:
  - image_generation
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
---


## Intro

최근 이미지 생성 관련 업무를 하면서 고민했던 문제가 있습니다. 배경 이미지에 어떤 물체를 추가해야 하는 상황이었고 이때 단순히 물체를 attach 시키는 것은 물론 추가된 물체가 배경에 잘 녹아들 수 있어야 했습니다.
여기서 포인트는 배경과 물체에 대한 이미지가 이미 있다는 가정입니다. 이미지를 처음부터 생성해야 했다면 Multi Area Conditioning으로 배경과 물체를 한번에 생성할 수 있지만 보유하고 있는 물체 이미지를 활용해야만 했습니다.

그래서 여러 고민과 실험끝에 선택한 방법에 대해 이번 포스팅에서 공유하는 시간을 가져보려고 합니다.

<br>

## Dive in

"특정 물체를 원하는 이미지에서 원하는 위치에 추가한다." 라는 문제를 해결하기 위한 방법으로 크게 두가지 방법을 생각했습니다.

**1. object 이미지를 IP-Adapter 모델에 입력하여 배경이미지의 마스킹한 위치에 녹여낸다.**  

- 장점:  
  - 배경과 물체가 하나가되어 매우 자연스럽게 이미지가 생성된다.

- 단점:    
  - 물체와 배경이 서로 조합되는 과정에서 물체의 위치가 마스킹한 위치와 조금 달리질 수 있다. -> 정확한 위치 설정이 어려움
  - 물체가 100% 재현되지 않고 약간이라도 물체가 변한다. -> 물체가 유지되어야 하는 경우 매우 치명적임

<br>

**2. object 이미지를 단순히 paste한 뒤에 controlnet 으로 배경이미지의 마스킹한 위치만 재생성한다.**

- 장점:  
  - 물체를 지정한 위치에 정확하게 생성 가능하다.
  - object를 100% 유지할 수 있으며 필요시 배경과 자연스럽게 스타일을 변환할 수 있다.

- 단점:
  - 기본적으로 배경위에 물체가 덮어지는 방식이므로 배경과 어울리지 않을 수 있다.
  


위 방법의 장,단점을 도메인의 요구사항과 비교했을 때 더 나은 방법인 2번 방법을 선택하였습니다.

<br>

### Workflow

![](/images/object_combine_workflow.png){: .align-center}

workflow는 아래와 같은 순서로 크게 5개의 노드 그룹으로 구성하였습니다.

1. 배경 이미지위에 마스킹
2. object 이미지 배경 제거 후 배경에 attach
3. controlnet을 활용한 resampling
4. post processing
5. upscaling

<br>

#### 1. Masking on background

![](/images/object_combine_node1.png){: .align-center}

> Model Load

저는 Base 모델로 [dreamshaper_sdxl_lightning](https://civitai.com/models/112902?modelVersionId=354657){:target="_blank" style="color: red;" } 모델을 사용하였습니다. sdxl lightning은 24년 2월에 발표된 모델로 sdxl turbo와 lcm과 같이 4~6의 적은 step으로 1024x1024 수준의 이미지를 빠르게 이미지를 생성할 수 있습니다.

추가로 디테일을 보안하기 위해 [detail tweaker](https://civitai.com/models/122359?modelVersionId=135867){:target="_blank" style="color: red;" } 라는 sdxl lora 모델을 같이 사용했습니다.

<br>

> Image Load &  Masking

이미지를 로드해준 뒤에 ComfyUI 자체 마스킹 툴로 좌측 하단에 마스킹을 합니다. 마스킹이 완료되면 Mask To Region 노드를 통해 drawing으로 이루어진 마스크를 사각형 마스크로 변환합니다. 이 작업이 필요한 이유는 물체가 잘리는 현상을 방지하기 위함입니다.

<div style="display: grid; grid-template-columns: repeat(2, 2fr); grid-gap: 10px;">
<div>
<img src="/images/object_combine_node1-2.png" alt="Image 1" style="max-width: 100%; height: auto;">
<p style="text-align: center;">drawing mask</p>
</div>
<div>
<img src="/images/object_combine_node1-1.png" alt="Image 2" style="max-width: 100%; height: auto;">
<p style="text-align: center;">square mask</p>
</div>

</div>

drawing mask를 사용하게 되면 그 마스크 안에 물체가 들어오기 때문에 위 왼쪽 그림처럼 캐릭터의 머리카락이 조금 잘리는 것을 볼 수 있습니다. 이러한 현상을 방지하기 위해 drawing 마스크의 최대 크기의 square 마스크로 변환하였습니다.

<br>

> Prepare Image & Mask for Inpaint

이미지에서 특정 영역만 변환할 때 만약 그 영역이 매우 작으면 어떻게 될까요? 이러한 행동은 1024x1024 수준에서 최고의 퍼포먼스를 보이는 모델에게 200x200의 입력을 주는 꼴이 됩니다. 당연하게도 생성된 이미지의 퀄리티는 매우 낮을 것입니다.

이러한 문제를 해결하기 위해 Prepare Image & Mask for Inpint 노드를 활용하였습니다.

![](/images/object_combine_node1-3.png){: .align-center}

Prepare Image & Mask for Inpint 노드는 원본 이미지의 비율을 유지한 상태로 마스크 주변 영역을 crop해 줍니다. 결국 배경 이미지에서 모델이 집중해야할 필요가 있는 부분만 추출해 내는 것 입니다. 뒤에서 inpaint_image의 inpaint_mask 영역에 캐릭터(물체)가 들어가게되며 그것을 다시 overlay_image의 영역으로 돌아가게 됩니다.

이러한 작업은 매우 큰 사이즈의 이미지에 대한 작업이 필요할 때에도 도움이 됩니다. 이미지 사이즈가 큰 경우 단순히 downscale하여 모델에 적용하는 것 보다 원하는 영역에서만 작업하고 원본 이미지에 붙이기만 한다면 훨씬 더 효율적이면서 원본 고화질 이미지 손실을 최소화할 수 있습니다. 

<br>

#### 2. Background Remove and Paste

![](/images/object_combine_node2.png){: .align-center}


> Remove Background

object 이미지에 배경이 없다면(transparency) 가장 좋지만 현실적으로 그런 이미지를 구하기도, 만들어내기도 번거롭습니다. 그래서 어떠한 물체 이미지를 입력하더라도 자동으로 배경을 지워줄 수 있는 노드를 추가하였습니다.

Background Remover로 가장 많이 알려진 모델은 [removebg](https://www.remove.bg/){:target="_blank" style="color: red;" } 입니다. API도 존재하며 오픈소스 모델로 파이썬 코드나 ComfyUI의 노드로 활용할 수 있습니다. 하지만 공개된 removebg 모델의 성능이 아쉬워서 여러가지 모델을 테스트 하다가 [BRIA](https://huggingface.co/briaai/RMBG-1.4){:target="_blank" style="color: red;" } 라는 모델의 성능이 괜찮아서 해당 모델을 사용했습니다.

<br>

> Paste By Mask

위 Prepare Image & Mask for Inpaint 노드에서 crop한 이미지와 마스크에 배경이 제거된 object를 paste하는 노드입니다. 

resize_behavior라는 옵션에서 ["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"] 중 하나의 방식으로 paste를 진행할 수 있습니다. 저는 object의 비율을 유지한 상태로 마스크영역에 넣기 위헤 "keep_ratio_fit"을 선택하였습니다.

<br>

#### 3. Resampling with Line-art ControlNet 

![](/images/object_combine_node3.png){: .align-center}

> ControlNet

ControlNet은 어떤 가이드라인에 따라 이미지를 생성하는 기술입니다. 여기서 말하는 가이드라인은 canny, lineart, depth, pose 등 여러가지 종류가 있습니다. 여러 방법과 모델을 실험한 끝에 현재 task에 가장 준수했던 조합은 lineart 처리 + [t2i-adapter-sdxl-sketch](https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0){:target="_blank" style="color: red;" } 모델 입니다.

Realistic Lineart 노드를 통해 lineart 이미지를 얻고 Apply ControlNet 노드에 입력해 줍니다. 그 다음 KSampler를 통해 이미지를 생성하기전에 latent_image를 준비해야 합니다.

latent_image를 입력하여 단순히 "sampling"이 아닌 위에서 Paste By Mask 노드를 통해 만든 이미지로  "resampling"해야 하는 이유는 뭘까요?

<div style="display: grid; grid-template-columns: repeat(2, 2fr); grid-gap: 10px;">
<div>
<img src="/images/object_combine_node3-2.png" alt="Image 1" style="max-width: 100%; height: auto;">
<p style="text-align: center;">Empty Latent(denoise: 1.0)</p>
</div>
<div>
<img src="/images/object_combine_node3-1.png" alt="Image 2" style="max-width: 100%; height: auto;">
<p style="text-align: center;">Latent Noise Mask(denoise: 0.55)</p>
</div>

</div>

왼쪽 그림은 랜덤노이즈로부터 Lineart 이미지만 참고하여 처음부터 생성하기 때문에 기존 이미지에 대한 고려를 하지 않아 전혀 다른 느낌으로 이미지가 생성되는 것을 볼 수 있습니다.

반면에 Paste By Mask 노드의 output 이미지를 기반으로 생성한 오른쪽 이미지는 0.55의 denoise 수치에서 기존 이미지의 스타일을 유지하는 것을 볼 수 있습니다. denoise 수치를 높일수록 이미지가 많이 변하게 됩니다.

<br>

#### 4. Post processing

![](/images/object_combine_node4.png){: .align-center}

> ImageCompositeMasked

위 resampling을 거쳐 생성된 이미지를 바로 원본 이미지의 위치로 보내게 되면 문제가 발생합니다. 이미지가 resampling 되면서 물체 주변의 영역이 일부분 원본과 달라지기 때문입니다.

![](/images/object_combine_node4-1.png){: .align-center height="50%" width="50%"}

저희의 목적은 "object"만 붙여넣는 것이었기 때문에 배경이 변해서는 안됩니다. 그래서 저는 여기에 다시한번 background remover와 ImageCompositeMasked 노드를 사용하였습니다.

우선 BRIA RMBG 노드를 통해 생성된 이미지에서 object에 대한 mask를 가져옵니다. 그리고 Prepare Image & Mask for Inpaint 노드의 inpaint image에 생성된 이미지의 물체만 attach 시켜줍니다.

이렇게 되면 원본 배경은 유지한채로 변화시킨 object만 붙여넣을 수 있게 됩니다.

<br>

> Overlay Inpainted Image

이제 원본 이미지의 영역에 되돌려보낼 시간입니다. Overlay Inpainted Image 노드는 Image & Mask for Inpaint 노드와 한 쌍으로 이루어져 있습니다. Image & Mask for Inpaint 노드의 output인 overlay_image와 crop_region을 그대로 가져오고 ImageCompositeMasked 노드의 output을 연결해주면 resampling으로 생성한 object가 추가된 원본 스케일 이미지를 얻을 수 있습니다.

<br>

#### 5. Upscaling

![](/images/object_combine_node5.png){: .align-center}

> Ultimate SD Upscale

이미지의 퀄리티를 한층 더 높이기 위해 Ultimate SD Upscale 노드를 추가하였습니다. Ultimate SD Upscale 노드의 특징은 upscale 모델과 diffusion 방식의 upscale의 조화입니다.

upscale 모델은 많이 사용되는 [4x_foolhardy_Remacri](https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/tree/main){:target="_blank" style="color: red;" } 를 사용하였습니다.

Ultimate SD Upscale는 tile 기법 즉, 이미지를 패치로 나누어 이미지를 훑어가며 upscale을 진행합니다. 이것의 장점은 아주 큰 사이즈의 이미지에서도 제한된 메모리로 동작시킬 수 있다는 것입니다. 다만 이런 경우 시간이 많이 소요될 수 있습니다.

나머지 세팅은 기본값으로 두고 생성한 결과는 아래와 같습니다.

<div style="display: grid; grid-template-columns: repeat(3, 3fr); grid-gap: 10px;">
<div>
<img src="/images/object_combine_node5-1.png" alt="Image 1" style="max-width: 100%; height: auto;">
<p style="text-align: center;">original background</p>
</div>
<div>
<img src="/images/object_combine_node5-2.png" alt="Image 2" style="max-width: 100%; height: auto;">
<p style="text-align: center;">combine with keep object</p>
</div>
<div>
<img src="/images/object_combine_node5-3.png" alt="Image 2" style="max-width: 100%; height: auto;">
<p style="text-align: center;">combine with resampled object</p>
</div>
</div>


<br>

## END

저는 평소에 ComfyUI를 통해 새로운 기술을 조합하여 워크플로우를 하나의 feature로 만들어내고 코드로 변환하여 서비스에 배포시키고 있습니다. 처음으로 ComfyUI에 대해 포스팅을 작성했는데 이미지만 캡쳐하다가 시간을 다 쓴것 같네요 ㅎㅎ; 

앞으로도 시간 날때마다 ComfyUI로 작업하면서 재미있는 실험이나 겪었던 문제에 대해서 공유하는 시간을 가져보겠습니다. 감사합니다.   