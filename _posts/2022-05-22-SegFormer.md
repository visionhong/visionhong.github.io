---
title:  "[논문리뷰] SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
folder: "computer_vision"
categories:
  - computer_vision
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
---

이번 포스팅은 HONG KONG University와 NVIDIA에서 2021년 10월에 발표한 SegFormer라는 논문을 리뷰하려고 한다. SegFormer는 이름에서부터 알 수 있듯이 Transformer를 Semantic Segmentation task에 적용한 모델이다.(최초 시도는 아님)

![](/images/../images/2023-03-12-02-02-04.png){: .align-center height="70%" width="70%"}

저자들은 위 그래프를 통해 SegFormer모델이 파라미터 수 대비 모델의 정확도(IoU)가 Efficient 하다는 것을 강조하고 있다. 어떤 방법으로 좋은 성능을 낼 수 있었는지 SegFormer에 대해 자세히 알아보자.

---

#### **Abstract**

SegFormer는 다음 두가지 특징을 가지고 있다.

1) SegFormer는 multiscale feature를 output으로 뽑는 계층적 구조의 Transformer encoder로 구성된다. Transformer encoder에서는 Transformer 구조에서 각 패치의 위치 정보를 위해 사용하는 positional encoding이 필요없도록 하여 이에따라 학습에 사용되지 않은 이미지 사이즈를 테스트시에 사용했을때 interpolation 사용으로 인한 성능 하락을 피할 수 있었다.

2) 복잡한 decoder를 고려하지 않고 MLP로만 이루어진 MLP decoder를 사용하였다. encoder에서 얻은 multiscale feature를 결합하여 각 feature map에서의 local attention과 합쳐진 feature map에서의 global attention을 통해 powerful한 representation을 얻었다.

저자들은 간단하고 가벼우면서 효율적인 Transformer Segmentation이라는 것을 강조하고 있으며 모델의 스케일에 따라 SegFormer-B0 부터 SegFormer-B5 까지 제안하였다. (Cityscapes, ADE20K 데이터셋에서 매우 유의미한 성능을 보임)

---

#### **Method**

입력 이미지 사이즈가 H x W x 3 일때 VIT(Vision Transformer)에서는 patch size를 16 x 16으로 설정하였는데 저자들은 dense prediction task에서 더 높은 성능을 뽑기 위해 그보다 더 작은 4 x 4 path size를 사용하였다. 이렇에 나눈 patch들은 multi-level feature map 을 뽑아내는 Transformer encoder로 들어가게 되며 이때 각 feature map의 size는 원본 이미지에 대해 {1/4, 1/8, 1/16, 1/32}으로 설정하였다. 

MLP decoder에서는 multi-level feature map을 여러 레이어에 거쳐 최종적으로 H/4 x H/4 x N(cls) resolution을 갖는 Segmentation mask를 예측하게 된다. Encoder와 Decoder에 대해 자세히 살펴보자.

![](/images/../images/2023-03-12-02-02-14.png){: .align-center height="70%" width="70%"}

<br>

**1\. Hierarchical Transformer Encoder**

저자들은  SegFormer의 Encoder를 Mix Transformer Enoder(MiT)라고 부르며 ViT에서 받은 영감을 Semantic Segmentation에 적용시켰다. MiT는 B0부터 B5까지 여러 사이즈로 모델이 구분되어 있으며 MiT-B5가 가장 크면서 가장 좋은 성능을 내는 모델이다.

<br>

**Hierarchical Feature Representation**

single-resolution feature map을 생성하는 ViT와 다르게 SegFormer에서는 high-resolution coarse feature와 low-resolution fine-grined feature를 가지는 multi-level feature map을 통해 Semantic Segmentation 성능을 끌어 올렸다.

각 Stage(4단계)별로 feature map은 아래와 같은 방식으로 resolution과 Channel을 갖는다.

![](/images/../images/2023-03-12-02-02-27.png){: .align-center}

<br>

**Overlapped Patch Merging**

ViT에서는 N x N x 3 patch를 1 x 1 x C 벡터로 표현하였다. 이때 각 patch들은 서로 non-overlap 상태이기 때문에 patch들 간에 local continuity 가 보존되기가 어렵다. 이를 해결하기위해 Swin Transformer에서는 Shifted Window를 통해 패치들간의 local continuity를 보존하려고 했고 SegFormer에서는 이 문제를 overlapping patch merging 으로 접근하였다.

단순히 4x4 patch size로 나누어 vector embedding을 진행하는 것이 아니라 마치 CNN이 sliding window로 조금씩 겹쳐가면서 연산을 진행하는것과 같이 K(patch size or kernel size), S(stride), P(padding)를 사전에 정의하여 B(batch) x C(channel x stride^2) x N(num of patch) 의 차원으로 patch를 분할하고 B(batch) x C(embedd dim) x W(width) x H(height) 의 차원으로 Merging을 수행한다.

<br>

**Efficient Self-Attention**

Encoder의 self-attention layer는 많은 연산량을 차지한다는 문제점이 있다. 게다가 SegFormer는 patch size가 16x16 이 아닌 4 x 4이기 때문에 더 많은 파라미터를 연산할 수 밖에 없다. 기존 multi-head self-attention 프로세스는 Q(query), K(key), V(value)를 모두 N(H x W) x C 차원을 가지는 행렬로 만들어 아래 식으로 계산이 되었다.

![](/images/../images/2023-03-12-02-02-41.png){: .align-center height="50%" width="50%"}

위 수식은 O(N^2) 계산복잡도를 가지기 때문에 large image가 입력으로 들어오게 된다면 모델이 급격하게 무거워진다. 그래서 저자들은 reduction ratio를 사전에 정의하여 K와 V의 N(H x W)채널을 줄이는 sequence reduction process를 적용하였다. 

![](/images/../images/2023-03-12-02-02-52.png){: .align-center height="40%" width="40%"}

위 수식과 같이 N을 R로 나누고 C에 R을 곱하면 Reshape이 가능해지고 이때 C X R을 Linear 연산을 통해 다시 C로 줄임으로써 N/R x C차원으로 Key와 Value로 만들어 줄 수 가 있다. 저자들은 실험을 통해 Stage-1부터 Stage-4 까지의 R을 \[64, 16, 4, 1\]로 설정하였다.

<br>

**Mix-FFN**

ViT에서는 지역 정보를 추가하기 위해 positional encoding을 적용시켰다. 하지만 이런 방식은 input resolution이 고정되어야 한다는 문제가 있으며 이는 input resolution이 달라지게 되면 interpolation을 통해 크기를 맞춰줘야 해서 성능 하락을 유발한다. 이에 저자들은 positional encoding이 semantic segmentation에 꼭 필요한 것은 아니라고 하며 positional encoding을 대신하여 3 x 3 Convolution (stride: 1 / padding: 1)을 FFN에 적용시켰다. (3 x 3  Conv의 zero padding을 통해 leak location의 정보를 고려할 수 있다고 주장)

![](/images/../images/2023-03-12-02-03-04.png){: .align-center height="50%" width="50%"}

이에 대해 수식은 위와같이 기존 Transformer encoder의 FFN(Feed Forward Network)에서 Conv3x3 layer만 추가되었다. 실험을 통해 3 x 3 convolution이 충분히 Transformer에 위치 정보를 제공할 수 있다는 것을 보였고 파마미터 수를 줄이기 위해 3x3 convolution을 depth-wise convolution으로 사용하였다.

<br>

**2\. Lightweight All-MLP Decoder**

저자들은 MLP layer로만 구성되어 있는 lightweight decoder를 설계하였고 이는 기존 다른 모델의 decoder 와는 다르게 수작업 및 연산량이 크게 요구되지 않는다는 것을 강조한다. 이렇게 간단한 decoder를 설계를 할 수 있었던 이유는 hierachical transformer encoder에서 larger effective field를 가질 수 있었기 때문이라고 한다. MLP Decoder는 아래 방식으로 순차적으로 진행이 된다.

![](/images/../images/2023-03-12-02-03-15.png){: .align-center height="50%" width="50%"}

1\. multi-level feature들의 channel을 모두 동일하게 통합시킨다.

2\. feature size를 original image의 1/4 크기로 통합한다.

3\. feature들을 concatenate시키고 이 과정에서 4배로 증가한 channel을 원래대로 돌린다.

4\. 최종 segmentation mask를 예측한다. (shape: B(batch) x N(num of class) x H/4 x W/4)  

---

#### **Experiments**

![](/images/../images/2023-03-12-02-03-31.png){: .align-center height="70%" width="70%"}<br>

![](/images/../images/2023-03-12-02-03-37.png){: .align-center height="70%" width="70%"}<br>

![](/images/../images/2023-03-12-02-03-46.png){: .align-center height="70%" width="70%"}


---

#### **Conclusion**

저자들은 Positional-encoding free, hirerachical Transformer encoder, lightweight All-MLP decoder 기법을 가지는 SegFormer라는 Semantic Segmentation 모델을 제안하였다. 복잡한 디자인을 가지는 기존의 방법을 피하고 efficiency하면서 좋은 performance를 이끌어 내었다. 단순히 여러 Dataset에서 SOTA를 달성한것 뿐 아니라 stron zero-shot robustness를 보여주었다.

단 한가지 한계점은 SegFormer의 small model인 SegFormer-B0의 파라미터 수는 3.7M 개로 일반적인 CNN 모델보다 적은 수 이지만 100k memory수준의 가벼운 edge device에서 잘 작동 될지는 불분명하다고 하며 이 부분은 지속적으로 연구가 되어야 한다고 말한다.

---

#### **PyTorch Implementation**

![](/images/../images/2023-03-12-02-06-34.png){: .align-center height="70%" width="70%"}

-   Pi: the padding size of the overlapping patch embedding in Stage i;
-   Ci: the channel number of the output of Stage i;
-   Li: the number of encoder layers in Stage i;
-   Ri: the reduction ratio of the Efficient Self-Attention in Stage i;
-   Ni: the head number of the Efficient Self-Attention in Stage i;
-   Ei: the expansion ratio of the feed-forward layer in Stage i;
-   Ki: the patch size of the overlapping patch embedding in Stage i;
-   Si: the stride of the overlapping patch embedding in Stage i;
-   Ki: the patch size of the overlapping patch embedding in Stage i;

위 표에서 B0모델의 R을 보면 Stage별로 \[8, 4, 2, 1\]로 되어있다. R은 Reduction ratio로 Efficent Self-Attention에 사용될 값인데 Method에서 소개할때는 \[64, 16, 4, 1\]이라고 했지만 표에선 다르게 적힌 이유는 self attention을 진행하기 전 matrix 차원을 N(H x W) x C로 나타내는데 Stage-1의 값으로 예를 들면 8은 H 혹은 W의 reduction ratio이고 64는 N의 reduction ratio를 의미하므로 결국 같은 의미이다.

<br>

**SegFormer-B0**

``` python
import torch
from torch import nn, einsum
from einops import rearrange
from math import sqrt

class DsConv2d(nn.Module):
    '''
    Mix-FFN에 Positional encoding을 대신할 3x3 Depthwise separable convolution
    '''
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super(DsConv2d, self).__init__()
        self.net = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_in,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=dim_in,
                      bias=bias
                      ),
            # Pointwise Convolution
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=1,
                      bias=bias
                      )
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    '''
    Efficient Self-Attn block과 Mix-FFN block 전에 사용될 Layer Normalization
    '''
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()  # std = root(variation)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))
```

```python
class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super(EfficientSelfAttention, self).__init__()
        self.scale = (dim // heads) ** -0.5  # root(dim head)
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim*2, reduction_ratio, stride=reduction_ratio, bias=False)  # ESA
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))  # chunk: dim=1을 기준으로 2개로 쪼갬
        # q,k,v의 차원을 모두 아래와 같이 변경
        # batch, (head*channel), w, h -> (batch*head), (w*h), channel
        # 첫 stage에서 16384 -> 256 으로 행렬곱 연산을 64배 감소시킴(Efficient multi head self attention)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        # einsum을 아래와 같이 행렬곱으로 사용 가능, q@k로도 가능
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # Q*K / root(dim head)
        attn = sim.softmax(dim=-1)  # dim=-1 -> channel

        out = einsum('b i j, b j d -> b i d', attn, v)  # 최종 attention 계산
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)  # 차원 복구
        return self.to_out(out)
```

``` python
class MixFeedForward(nn.Module):
    def __init__(self, *, dim, expansion_factor):
        super(MixFeedForward, self).__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # positional embedding 대체
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)
```

``` python
class MiT(nn.Module):
    '''
    Mix Transformer encoders
    '''
    def __init__(self,
                 *,
                 channels,
                 dims,
                 heads,
                 ff_expansion,
                 reduction_ratio,
                 num_layers
                 ):
        super(MiT, self).__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)  # *: take off tuple  (3, 32, 64, 160, 256)
        dim_pairs = list(zip(dims[:-1], dims[1:]))  # [(3, 32), (32, 64), (64, 160), (160, 256)]

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            # Layer Norm -> ESA -> Layer Norm -> MFFN
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
               get_overlap_patches,
               overlap_patch_embed,
               layers
            ]))

    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)  # (b, c x kernel x kernel, num_patches)

            num_patches = x.shape[-1]
            ratio = int(sqrt(h * w / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)  # (b, c(cxkernelxkernel), h, w)
            x = overlap_embed(x)  # (b c(embed dim) h w)
            for (attn, ff) in layers:  # attention, feed forward
                x = attn(x) + x  # skip connection
                x = ff(x) + x

            layer_outputs.append(x)  # multi scale features

        return x if not return_layer_outputs else layer_outputs
```

-   overlapped patch를 생성할 때 torch.nn.Unfold 함수를 사용하였는데 직관적으로 이해를 돕기위해 아래 그림으로 그려보았다.
-   Unfold함수는 반드시 4차원(b x c x w x h)의 데이터를 input으로 받고 3차원(B x C(channel x kernel^2) x N(num of patches))의 output을 낸다.

![](/images/../images/2023-03-12-02-07-13.png){: .align-center height="70%" width="70%"}

``` python
class SegFormer(nn.Module):
    '''
    Default values from Mix Transformer B0
    '''
    def __init__(self,
                 *,
                 dims=(32, 64, 160, 256),
                 heads=(1, 2, 5, 8),
                 ff_expansion=(8, 8, 4, 4),
                 reduction_ratio=(8, 4, 2, 1),
                 num_layers=(2, 2, 2, 2),
                 channels=3,
                 decoder_dim=256,
                 num_classes=4):
        super(SegFormer, self).__init__()
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )
        # 논문 모델 그림에서 MLP Layer
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),  # First, multi-level features Fi from the MiT encoder go through an MLP layer to unify the channel dimension.
            nn.Upsample(scale_factor=2**i)  # second step, features are up-sampled to 1/4th and concatenated together.
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),  # Third, a MLP layer is adopted to fuse the concatenated features
            nn.Conv2d(decoder_dim, num_classes, 1),  # Finally, another MLP layer takes the fused feature to predict the segmentation mask M with a H4 × W4 × Ncls resolution
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)
```

``` python
if __name__ == '__main__':
    model = SegFormer()
    img = torch.rand((2, 3, 512, 512))

    print(f'OUTPUT Shape: {model(img).shape}')
```

OUTPUT Shape: torch.Size(\[2, 4, 128, 128\])

---

Reference

Paper - [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)  
Code - [https://github.com/tjems6498/Vision/blob/master/Semantic\_Segmentation/SegFormer/segformer.py](https://github.com/tjems6498/Vision/blob/master/Semantic_Segmentation/SegFormer/segformer.py)