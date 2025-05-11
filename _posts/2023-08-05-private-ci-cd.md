---
title:  "Private(On-premise) 환경에서 CI/CD 구축하기"
folder: "tools"
categories:
  - tools
tags:
  - MLOps
  - Backend
  - Github Actions
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
header:
  teaser: "/images/flops-1.png"
---

## Intro

안녕하세요. 최근 개인적으로 Federated Learning 환경을 구축 할 일이 있었는데 그 중 CI/CD 부분에 대해 포스팅을 하려고 합니다.

Federated Learning이란 하나의 글로벌 서버가 있고 여러 클라이언트(웹, 모바일 등)가 자신의 데이터를 자신의 리소스를 사용하여 학습을 진행하고 학습된 모든 클라이언트의 가중치를 종합하여 글로벌 서버의 모델을 업데이트 하는 학습방법론 입니다.

Federated Learning은 의료 헬스케어, 금융, IoT 등 데이터가 외부로 노출되면 안되는 분야에서 사용될 수 있습니다. 

예를 들어 어떤 병원에서 환자 데이터 활용해 AI모델을 개발한다고 가정해 보겠습니다. 데이터는 보안상 외부로 보낼 수 없기 때문에 모델을 학습하려면 그 병원 내의 데이터로만 학습해야 하기 때문에 성능을 높이는데 한계가 있습니다. 
하지만 Federated Learning을 적용한다면 다른 병원의 데이터로 학습된 모델의 가중치를 활용할 수 있기 때문에 보안을 유지한 채 조금 더 모델의 성능을 끌어올릴 수 있습니다.

이런 Federated Learning 환경에서도 코드 변경이 일어날 수 있기 때문에 CI/CD 를 통해 글로벌 서버를 지속적으로 배포할 수 있어야 합니다. Cloud 환경에서도 CI/CD를 구축할 수 있지만 저는 서버가 준비되어 있기 때문에 On-Premise에서 구축하였고 그것에 대해 공유하려고 합니다.

<br>

## Architecture

CI/CD 아키텍쳐는 아래와 같습니다.

Environment
- Ubuntu 20.04
- Kubernetes v1.27.4
- GitHub Actions (Self-hosted runner)
- Harbor: v2.8.2
- ArgoCD: v2.7.8

![](/images/flops-1.png){: .align-center height="85%" width="85%"}


Workflow에 대해 간단히 설명드리면 개발자가 코드를 수정하여 GitHub Code Repository에 push하면 Github Actions가 트리거되어 이미지를 빌드 및 푸시하고 Config Repository의 manifest file을 업데이트 시킵니다.

Config Repository를 바라보고 있던 ArgoCD는 변동을 자동 감지하여 새로운 버전의 글로벌 서버를 배포하게 되는 구조입니다.

각 과정에 대한 자세한 설명은 아래에서 말씀 드리겠습니다. (참고로 ArgoCD와 Harbor에 대한 설치 방법까지 다루기엔 글이 너무 길어질 것 같아 생략하였습니다. 두 오픈소스 설치 방법은 공식문서를 참조하시면 될 것 같습니다.)

<br>

### 툴을 선택한 이유


> CI

대표적인 CI툴로는 Jenkins가 있습니다. 하지만 Jenkins를 사용하기 위해서는 서버에 Jenkins를 설치를 해야하고 여러가지 설정을 해야 하기 때문에 복잡합니다. 이에 비해 GitHub Actions는 GitHub과 밀접하게 작동하기 때문에 workflow를 정의한 yaml 파일만 작성하면 CI를 구축할 수 있습니다. 

다만 기본적으로 Github Actions는 'Github-hosted runner' 즉 Github에서 제공하는 자체 호스트를 사용하여 동작하게 됩니다. 이때 Public Repository가 아니라면 일정 사용량을 넘으면 요금이 발생합니다. 

다행히 Private Repository라도 자체 호스트 러너(Self-hosted runner)를 사용한다면 무료로 사용할 수 있으며 Github-hosted runner에 비해 속도도 더 빠릅니다.

또한 Self-hosted runner를 사용하는 것이 아키텍쳐 구조상 보안적으로 좋았습니다. 추후에 나오는 github actions yaml 파일에는 docker login을 수행하는 부분이 있습니다. 기본적으로 docker login은 도커 허브와 https로 통신하기 때문에 Private registry인 Harbor에 https로 로그인 하기 위해서는 인증서가 필요합니다.

이때 Self-hosted runner를 사용하기 때문에 이미 발급받은 인증서가 서버내에 존재하므로 바로 Harbor에 로그인이 가능하게 됩니다. 만약 Github-hosted runner를 사용한다면 repository 안에 인증서를 저장해야 합니다. 아무리 Private repository 이더라도 관리자가 아닌 사람이 접근할 수 있어 보안적으로 좋지 않고 github actions yaml 파일 내용이 복잡해지기 때문에 Self-hosted runner를 선택하였습니다.

<br>

> CD

CD 툴로는 ArgoCD를 선택하였습니다. ArgoCD의 장점에 대해서는 [이전 포스팅](https://visionhong.github.io/tools/ArgoCD/#benefits-of-using-argocd){:target="_blank" style="color: brown;" } 에서 자세히 설명해 두었습니다.

간단히 정리하자면 글로벌 서버를 쿠버네티스 환경에서 CronJob으로 배포할 것이기 때문에 쿠버네티스 위에서 동작하는 ArgoCD를 바로 적용할 수있으며 별도의 자격증명 없이 CD를 운영하고 UI에서 관리 및 모니터링 할 수 있으면서 GitOps의 장점까지 가져갈 수 있어 CD 툴로 많이 사용되고 있습니다.

<br>

> Image Registry

이미지 저장소로는 Private Registry 오픈소스로 알려져 있는 Harbor를 선택했습니다.  On-premise에서 선택할 수 있는 저장소는 Docker hub, Github Container registry, Harbor 가 대표적입니다. 

Docker hub와 Guthub Container registry를 사용한다면 workflow에서 쉽게 다룰 수 있지만 이 역시 Private Image로 저장하려면 요금이 발생합니다. 그렇기 때문에 서버내에 Harbor를 직접 운영하는 방법을 선택하였습니다.

Harbor는 자체 UI를 보유하고 있기 때문에 이미지를 UI에서 관리할 수 있으며 유저생성 등 다양한 기능을 제공합니다.


<br>

## CI/CD 구축

먼저 GitHub Repository를 두 개 생성해야 합니다. 하나는 코드가 들어가는 저장소이고 나머지 하나는 K8s manifest 파일 저장소로 사용됩니다. 보안을 위해 두 Repository 모두 Private으로 생성하겠습니다.

![](/images/flops-2.png){: .align-center height="85%" width="85%"}


제 Code Repository에는 클라이언트와 글로벌 서버 python script와 각각의 Dockerfile 등의 파일을 넣어두었습니다. 클라이언트와 글로벌 서버의 코드가 다르기도 하고 클라이언트 컨테이너에 글로벌 서버 코드가 들어가게 되면 문제가 될 수 있기 때문에 따로 구분을 해두었습니다.

<br>

### Github Actions

Code Repository의 .github/workflows 폴더에 Github Actions의 workflow를 정의한 yaml 파일을 넣어두었습니다.

actions.yml

``` yaml
name: BuildAndPushImagesAndUpdateArgoCDConfig

on:
  push:
    branches:
      - main

env:
  DOCKER_REGISTRY: <Harbor 도메인 주소>
  IMAGE_REPO: <Harbor 프로젝트 이름>

jobs:
  ci_cd_pipeline:
    runs-on: self-hosted
    steps:
      - name: Get current time
        id: time
        run: echo "::set-output name=time::$(date +%Y-%m-%d-%H-%M-%S)"

      - name: Checkout Repository
        uses: actions/checkout@v3
        
      - name: Login to Docker Registry
        uses: docker/login-action@v1
        with:
          registry: {% raw %}${{ env.DOCKER_REGISTRY }}{% endraw %}
          username: {% raw %}${{ secrets.HARBOR_USERNAME }}{% endraw %}
          password: {% raw %}${{ secrets.HARBOR_PASSWORD }}{% endraw %}

      - name: Build and Push server to Harbor
        id: build_and_push_server
        run: |
          docker build -f Dockerfile.server -t {% raw %}${{ env.DOCKER_REGISTRY }}{% endraw %}/{% raw %}${{ env.IMAGE_REPO }}{% endraw %}/server:{% raw %}${{ steps.time.outputs.time }}{% endraw %} .
          docker push {% raw %}${{ env.DOCKER_REGISTRY }}{% endraw %}/{% raw %}${{ env.IMAGE_REPO }}{% endraw %}/server:{% raw %}${{ steps.time.outputs.time }}{% endraw %}
      - name: Build and Push client to Harbor
        id: build_and_push_client
        run: |
          docker build -f Dockerfile.client -t {% raw %}${{ env.DOCKER_REGISTRY }}{% endraw %}/{% raw %}${{ env.IMAGE_REPO }}{% endraw %}/client:{% raw %}${{ steps.time.outputs.time }}{% endraw %} .
          docker push {% raw %}${{ env.DOCKER_REGISTRY }}{% endraw %}/{% raw %}${{ env.IMAGE_REPO }}{% endraw %}/client:{% raw %}${{ steps.time.outputs.time }}{% endraw %}
          
      - name: Setup SSH
        uses: MrSquaare/ssh-setup-action@v1
        with:
          host: github.com
          private-key: {% raw %}${{ secrets.SSH_PRIVATE_KEY }}{% endraw %}

      - name: Clone Repository
        run: git clone git@github.com:visionhong/FLOps-config.git

      - name: Update YAML File
        run: yq -i '.spec.jobTemplate.spec.template.spec.containers[0].image = "{% raw %}${{ env.DOCKER_REGISTRY }}{% endraw %}/{% raw %}${{ env.IMAGE_REPO }}{% endraw %}/server:{% raw %}${{ steps.time.outputs.time }}{% endraw %}"' 'FLOps-config/dev/cronjob.yaml'
        
      - name: Push Changes to Repo
        run: |
          cd FLOps-config
          git add .
          git commit -m "Update images by GitHub Actions"
          git push git@github.com:visionhong/FLOps-config.git --all

```

- github actions는 main branch로 push가 일어나면 self-hosted runner로 job이 실행됩니다.
- Actions workflow 요약
	1. 현재 시간을 환경변수로 생성 (이미지 태그를 runner 실행 시간으로 사용)
	2. Code Repository clone
	3. Harbor Registry에 로그인
	4. 서버와 클라이언트 Dockerfile을 활용해 이미지를 Build & Push
	5. secrets에 저장된 비공개키 등록 (Private Repository를 clone하기 위함)
	6. Config Repository clone
	7. manifest의 이미지 버전 업데이트
	8. GitHub Remote Config Repository에 push

<br>

"secrets." 부분은 GitHub Actions에서 사용할 수 있는 Secret 정보입니다. github repository의 Settings -> Security -> Secrets and variables -> Actions에서 등록할 수 있습니다.

Code Repository Secrets에 비공개 키를 넣은 이유는 SSH 키는 사용자가 비공개키를 활용하여 공개키를 가진 서버에 접속하는 방식으로 동작하기 때문입니다.
즉 Code Repository의 Github Actions Runner 는 사용자가 되고 GitHub Remote Config Repository를 서버로 볼 수 있습니다.

공개키는 Github 계정 -> Settings -> Access -> SSH and GPG keys 에 등록할 수 있습니다.

<br>

추가로 Self-hosted runner 설정은 github repository의 Settings -> Code and automation -> Actions -> Runners 에서 추가할 수 있습니다.

New self-hosted runner 버튼을 클릭했을 때 나오는 가이드 대로 설정을 완료하시면 run.sh 파일을 실행하여 Status를 "Idle" 상태로 만들 수 있습니다.

다만 run.sh 파일이 실행중이 아니라면 Github Actions runner 가 중단됩니다. 하루 종일 터미널을 켜둘 수 없으니 tmux를 활용하여 돌리시거나 `nohup ./runsh &` 명령어로 백그라운드에서 돌리는 방법을 사용할 수 있습니다. (저는 후자로 실행했습니다.)

<br>

### ArgoCD

CI를 구축했으니 이제 CD를 완성해 보겠습니다. ArgoCD 에서는 Config Repository를 바라 볼 수 있어야 하기 때문에 SSH key를 발급해서 Private Repository 와  연결해 주어야 합니다.

ArgoCD UI의 Settings -> Repositories -> CONNECT REPO 를 클릭하시면 Repository를 등록할 수 있습니다. 
여기서도 ArgoCD 입장에서 Config Repository에 접근하는 것이기 때문에 비공개키는 ArgoCD에, 공개키는 GitHub에 등록합니다.

ssh 방법으로 잘 등록하시면 아래 그림처럼 Connection Successful 됩니다.

![](/images/flops-4.png){: .align-center height="85%" width="85%"}


<br>

이제 Config Repository의 변화를 감지할 수 있도록 Applications를 UI에서 생성하겠습니다.

![](/images/flops-5.png){: .align-center height="85%" width="85%"}

![](/images/flops-6.png){: .align-center height="85%" width="85%"}

![](/images/flops-7.png){: .align-center height="85%" width="85%"}



생성된 Applications를 클릭해 보면 ArgoCD가 제가 미리 담아두었던 manifest 파일(Cronjob, Service)을 클러스터에 잘 배포한 것 처럼 보입니다.

하지만 CronJob으로 실행된 Pod의 실제 Status를 확인해 보면 ImagePullBackOff 으로 되어 있는 것을 확인할 수 있습니다.

``` bash
kubectl get po -n ci-cd-test
```
result:

![](/images/flops-8.png){: .align-center height="85%" width="85%"}

<br>

그 이유는 바로 쿠버네티스에서도 기본적으로 이미지를 Docker Hub 에서 Pull 하도록 되어있기 때문입니다. 그렇기 때문에 저희는 docker-registry secret을 생성하여 이미지를 Harbor Registry에서 가져올 수 있도록 해주어야 합니다.


``` bash
kubectl create -n ci-cd-test secret docker-registry harbor --docker-server=<your-registry-server> --docker-username=<your-name> --docker-password=<your-pword> --docker-email=<your-email>
```

위 명령어에 자신의 Harbor Registry의 정보를 입력해 secret을 생성하고 CronJob manifest 파일에서 아래와 같이 imagePullSecrets를 추가하시면 정상적으로 작동하게 됩니다.

cronjob.yaml

``` yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: global-server-cronjob
spec:
  schedule: "*/5 * * * *" # 매 5분마다 실행
  jobTemplate:
    spec:
      template:
        metadata:
          labels:
            app: global-server
        spec:
          restartPolicy: OnFailure
          containers:
            - name: global-server
              image: '<your domain>/<project>/<image>:<tag>'
              ports:
                - containerPort: 8080
          imagePullSecrets:
            - name: harbor
```

이제 정상적으로 CronJob이 동작할 것 입니다.

<br>

### Demo

자 이제 CI/CD가 모두 구축되었습니다. 코드를 변경해서 push를 하면 어떤 일이 발생하는지 확인해보겠습니다.

<iframe
class="embed-video youtube lazyload"
src="https://www.youtube.com/embed/xxW5t5TjSWM"
title="YouTube video player"
frameborder="0"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen
></iframe>

<br>
영상에서 utils.py를 수정하여 commit 했을 때 새로운 버전의 이미지가 생성되고 글로벌 서버가 업데이트 되는 것을 보실 수 있습니다.

<br>

## END

이번 포스팅에서는 On-Premise 환경에서 오픈소스를 활용하여 Private 하게 CI/CD를 구축하는 방법에 대해 알아보았습니다.

Github Actions, ArgoCD 모두 DevOps 에서 많이 사용되는 툴 이지만 머신러닝 분야에서도 충분히 활용할 수 있다고 생각합니다.

CI/CD에 더하여 학습된 글로벌 모델을 MLflow와 같은 오픈소스에서 버전관리를 하고 실제 모델을 클라이언트가 아닌 서버에서 배포하여 API 형태로 클라이언트에게 제공하는 등 Federated Learning을 자동화한 FLOps(Federated Learning Operations)를 만들어 낸다면 데이터 보안이 필수적인 분야에서 훨씬 더 편하게 Federated Learning에 접근할 수 있을 것 같습니다.