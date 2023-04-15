---
title:  "MLOps를 위한 Kubernetes"
folder: "tools"
categories:
  - tools
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
---

본 포스팅은 MLOps를 위해 필요한 k8s 지식을 정리하는 것을 목적으로함

**MLOps에서 k8s가 필요한 이유.**

MLOps를 위해서는 크게 다음과 같은 작업이 필요하다.

1.  Reproducibility - 실행 환경의 일관성 & 독립성
2.  Job Scheduling - 스케줄 관리, 병렬 작업 관리, 유휴 자원 관리 
3.  Auto-healing & Auto-scaling - 장애 대응, 트래픽 대응 자동화

\-> 이것들을 Docker(Containerization), k8s(Container Orchestration)를 통해 해결할 수 있다.

![](/images/../images/2023-03-12-01-02-16.png){: .align-center height="70%" width="70%"}

<br>

#### **Kubernetes**

Identity - Container Orchestration

-   여러명이 함께 서버를 공유하며 각자 모델학습을 돌리고자 할때 스케줄을 미리짜거나 gpu자원을 매번 확인하거나 학습이 끝나면 사용했던 메모리를 정리를 하는 등의 번거로운 일을 매번 할 필요 없이 수많은 컨테이너와 인프라 환경을 정해진 룰에 따라서 알아서 관리해주는 기술

Concept - 선언형 인터페이스 & Desired State

-   마치 우리가 길찾기를 할때 목적지만 적으면 현재위치에서 목적지로의 최적의 경로를 알려주는것을 선언형 인터페이스라고 볼 수 있고 여기서 '최적의 경로를 찾아줘'가 Desired State가 된다. 


---

#### **1.  minikube, kubectl**

**minikube**

-   마스터 노드의 일부 기능과 개발 및 배포를 위한 단일 워커 노드를 제공해 주는 간단한 쿠버네티스 플랫폼 환경을 제공
-   [https://minikube.sigs.k8s.io/docs/start/](https://minikube.sigs.k8s.io/docs/start/)

<br>

**kubectl**

-   쿠버네티스 클러스터에 요청을 간편하게 보내기 위해서 사용되는 client 툴
-   [https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-linux/](https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-linux/)

<br>

**minikube start**

minikube를 docker driver를 기반으로 하여 시작

``` bash
minikube start --driver=docker
```
![](/images/../images/2023-03-12-01-02-31.png){: .align-center height="70%" width="70%"}

<br>

**minikube status**

정상적으로 생성되었는지 minikube의 상태 확인

``` bash
minikube status
```

![](/images/../images/2023-03-12-01-02-41.png){: .align-center}

<br>

**minikube delete**

쿠버네티스 환경 삭제

``` bash
minikube delete
```

![](/images/../images/2023-03-12-01-02-51.png){: .align-center}

---

#### **2\. Pod**

-   Pod는 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 컴퓨팅 단위
-   쿠버네티스는 도커 컨테이너단위가 아닌 Pod단위로 스케줄링, 로드밸런싱, 스케일링 등의 관리 작업을 수행(쿠버네티스에 어떤 애플리케이션을 배포하고 싶다면 최소 Pod 단위로 구성해야 한다는 의미)
-   하나의 Pod는 한 개의 Container 혹은 여러개의 Container 로 이루어짐(Pod 내부의 여러 Container는 자원을 공유)
-   Pod는 stateless 한 특징을 지니고 있으며, 언제든지 삭제될 수 있는 자원
-   여러 노드에 1개 이상의 Pod를 분산 배포/실행 가능 (Pod Replicas)
-    Pod를 생성할 때 노드에서 유일한 IP를 할당 (하나의 Pod는 하나의 서버로 볼 수 있으며 유일한 IP는 서버 분리 효과라고 생각할 수 있음) -> 이 IP는 클러스터 내부에서만 접근 가능하기 때문에 클러스터 외부 트래픽을 받기 위해서는 Service 혹은 Ingress 오브젝트가 필요!
-   Pod내의 여러 컨테이너 간의 통신은 localhost로 가능하며 이때 각 컨테이너는 각자 다른 포트를 가지고 있어야 함 (볼륨 공유 가능)

<br>
**Pod와 컨테이너 설계 시 고려할 점 (Pod: Container = 1:1 or 1:N 결정)**

1.  컨테이너들의 라이프 사이클이 같은가
    -   A라는 컨테이너는 application이고 B라는 컨테이너는 로그 수집기라고 가정하면 A컨테이너가 종료되었을때 B컨테이너는 실행의 의미가 없다. -> 컨테이너를 묶어두는 방법을 사용할 수 있음
2.  스케일링 요구사항이 같은가
    -   어떤 application이냐에 따라 요구되는 트래픽이 다르기 때문에 스케일에 대한 고민이 필요하다.
3.  인프라 활용도가 더 높아지는 방향으로
    -   쿠버네티스는 각 노드의 리소스 등 여러 상태를 고려해서 Pod가 어떤 노드에 배포되면 좋을지 결정하게 된다. Pod의 크기가 불필요하게 클 경우 자원이 낭비되기 때문에 여러개의 컨테이너를 여러개의 Pod로 분리시키는 등의 방법으로 인프라를 효율적으로 사용해야 한다.
    -   Pod와 컨테이너를 1:1로 기본 설계하고 특별한 사유가 있는 경우 1:N 구조를 고민하는것이 일반적

<br>
**Pod의 한계점**

1.  Pod가 나도 모르는 사이에 종료되었다면?
    -   Self-Healing 이 없음, Pod나 노드 이상으로 종료되면 끝
    -   ReplicaSet 오브젝트를 도입하여 사용자가 선언한 수만큼 Pod를 유지
2.  Pod IP는 외부에서 접근할 수 없고 생성할 때 마다 Pod의 고유 IP가 변경된다. 
    -   클러스터 외부에서 접근할 수 있는 고정적인 단일 엔드포인트가 필요
    -   Pod 집합을 클러스터 외부로 노출하기 위한 Service 오브젝트 도입

<br>
**Pod 생성**

간단한 Pod 예시

``` yaml
apiVersion: v1 # kubernetes resource 의 API Version
kind: Pod # kubernetes resource name
metadata: # 메타데이터 : name, namespace, labels, annotations 등을 포함
  name: counter
spec: # 메인 파트 : resource 의 desired state 를 명시
  containers:
  - name: count # container 의 이름
    image: busybox # container 의 image
    args: [/bin/sh, -c, 'i=0; while true; do echo "$i: $(date)"; i=$((i+1)); sleep 1; done'] # 해당 image 의 entrypoint 의 args 로 입력하고 싶은 부분
```

``` bash
vi pod.yaml
# i를 누른 후 위 코드 붙여넣고 esc -> :wq -> enter로 수정

kubectl apply -f pod.yaml
# minikube delete를 했더라면 먼저 minikube start --driver=docker 실행
```

kubectl apply -f <yaml-file-path>를 수행하면, <yaml-file-path>에 해당하는 kubernetes resource를 생성 또는 변경 할 수 있다.
-   kubernetes resource 의 desired state를 기록해놓기 위해 항상 YAML 파일을 저장하고, 버전관리하는 것을 권장
-   kubectl run 명령어로 YAML 파일 생성 없이 pod를 생성할 수도 있지만, 이는 k8s에서 권장하는 방식이 아님

<br>

**Pod 조회**

current namespace의 의 Pod목록을 조회

namespace : k8s에서 리소스를 격리하는 가상의 단위

- 하나의 namespace에 여러 pod가 포함될 수 있음
- kubectl config view --minify 로 current namespace가 어떤 namespace인지 확인 할 수 있다.
- 기본 namespace는 default namespace

<br>

``` bash
kubectl get pod
```

![](/images/../images/2023-03-12-01-03-13.png){: .align-center height="50%" width="50%"}

<br>

특정 namespace 혹은 모든 namespace의 pod 조회

``` bash
kubectl get pod -n kube-system
# kube-system namespace 의 pod 을 조회

kubectl get pod -A
# 모든 namespace 의 pod 을 조회
```

![](/images/../images/2023-03-12-01-03-43.png){: .align-center height="70%" width="70%"}

<br>

**Pod 로그**

``` bash
kubectl logs <pod-name>

kubectl logs <pod-name> -f
# <pod-name> 의 로그를 계속 보여준다.

kubectl logs <pod-name> -c <container-name>
# pod안에 여러 개의 container가 있는 경우

kubectl logs <pod-name> -c <container-name> -f
```

<br>

**Pod 내부 접속**

``` bash
kubectl exec -it <pod-name> -- sh
# kubectl exec --help 로 사용법 확인

kubectl exec -it <pod-name> -c <container-name> -- <명령어>
# pod안에 여러개의 container가 있는 경우 (docker exec와 비슷)
```

![](/images/../images/2023-03-12-01-04-01.png){: .align-center height="50%" width="50%"}

<br>

**Pod 삭제**

``` bash
kubectl delete pod <pod-name>

kubectl delete -f <YAML-파일-경로>
# yaml파일이 존재한다면 yaml 파일을 사용해 삭제 가능
```

<br>

**Pod 환경변수**

Pod에 환경변수를 선언해서 컨테이너 속에 정보를 전달시킬 수 있다.

``` yaml
# red-app.yaml

apiVersion: v1
kind: Pod
metadata:
  name: red-app
spec:
  containers:
  - name: red-app
    image: yoonjeong/red-app:1.0
    ports:
    - containerPort: 8080  # Dockerfile에 EXPOSE시킨 포트번호
    env:
    - name: NODE_NAME
      valueFrom:
        fieldRef:
          fieldPath: spec.nodeName
    - name: NAMESPACE
      valueFrom:
        fieldRef:
          fieldPath: metadata.namespace
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
    resources:
      limits:
        memory: "64Mi"
        cpu: "250m"
```

``` bash
kubectl apply -f red-app.yaml
kubectl exec red-app -c red-app -- printenv POD_IP NAMESPACE NODE_NAME # 특정 컨테이너의 환경변수 확인
```

<br>

**Label과 Selector**

Label

-   쿠버네티스 오브젝트를 식별하기 위한 key/value 쌍의 메타정보
-   쿠버네티스 리소스를 논리적인 그룹으로 나누기 위해 붙이는 이름표

<br>

Selector

-   Label을 이용해 쿠버네티스 리소스를 필터링하고 원하는 리소스 집합을 구하기 위한 label query

Label을 이용해서 리소스 집합을 구해야하는 경우가 많고 k8s에서는 Label을 이용해서 명령을 실행할때 pod 집합 단위로 수행하게 된다. 그렇기 때문에 Pod를 생성할때 Label을 잘 설계하여 명시하는 것이 권장된다.

<br>

kubectl 명령어

``` bash
# Label 확인 (조건없이 확인만 할 때는 --show-labels만 작동)
kubectl get pods --show-labels <label-key>
kubectl get pods -L <label-key>

# Label 추가 (옆으로 연속 추가 가능)
kubectl label pod <pod-name> <label-key>=<label-value>

# Label 변경
kubectl label pod <pod-name> <label-key>=<label-value> --overwrite

# Label 삭제
kubectl label pod <pod-name> <label-key>-

# Pod조회 with Selector
kubectl get pod --selector <label query>
kubectl get pod -l <label query>

# ======================================
# label query 종류
# "key=value"
# "key!=value"
# "key in (value1, value2, ...)"  # or 연산
# "key notin (value1, value2, ...)"
# "key"  # label에 key가 있을때
# "!key"  # label에 key가 없을때
```

---

**ReplicaSet**

내결함성(fault tolerance)

\-> 소프트웨어나 하드웨어 실패가 발생하더라도 소프트웨어가 정상적인 기능을 수행할 수 있어야 한다.

\-> 사람의 개입없이 내결함성을 가진 소프트웨어를 구상해볼 수 있다.

<br>

**ReplicaSet 개념**

-   Pod 복제본을 생성하고 관리
-   ReplicaSet 오브젝트를 정의하고 원하는 Pod의 개수를 replicas 속성으로 선언
-   클러스터 관리자 대신 Pod 수가 부족하거나 넘치지 않게 Pod 수를 조정(replicas 속성으로 선언한 개수만큼 유지)

<br>

**ReplicaSet 역할**

-   ReplicaSet을 이용해 Pod 복제 및 복구 작업을 자동화
-   클러스터 관리자는 ReplicaSet을 만들어 필요한 Pod의 개수를 쿠버네티스에게 선언
-   쿠버네티스가 ReplicaSet 요청서에 선언된 replicas를 읽고 그 수만큼 Pod 실행 보장

<br>

ReplicaSet에 Replicas, Pod Selector, Pod Template을 정의해주면 수동적으로 pod yaml을 작성해서 배포할 필요가 없음

\-> 자동으로 pod 생성

ReplicaSet에 Port-Forward를 하게되면 첫번째 생성된 Pod로만 요청이 전달된다.

\-> 로드밸런싱이 일어나지 않음

<br>

kubectl 명령어

``` bash
# ReplicaSet과 배포 이미지 확인
kubectl get rs <replicaset-name> -o wide

# ReplicaSet의 Pod 생성 기록 확인
kubectl describe rs <replicaset-name>

# ReplicaSet의 Pod 생성 이후 과정 확인
kubectl get events --sort-by=.metadata.creationTimestamp

# ReplicaSet Pod로 트래픽 전달
kubectl port-forward rs/<replicaset-name> 8080:8080
```

<br>

**기존에 생성한 Pod를 ReplicaSet으로 관리하는 방법**

-   ReplicaSet은 자신이 관리하는 Pod의 수를 선언된 replicas를 넘지 않게 관리한다.
-   이때 기존에 Pod가 따로 생성되어있고 같은 label로 Pod Selector를 사용해 replicas=3으로 ReplicaSet을 생성하면 2개의pod만 추가적으로 생성한다.
-   즉 기 생성된 Pod의 label이 ReplicaSet의 Pod Selector와 같다면 관리 범주에 들어오므로 Pod Selector를 설계할 때 주의해야 함

<br>

**ReplicaSet이 내결함성(fault-tolerance)을 어떻게 지키는지 확인하기**

-   ReplicaSet이 관리하는 Pod를 삭제하면 새로운 Pod가 replicas 수만큼 자동생성됨

비슷해보이는 두가지 ReplicaSet 삭제방법

``` bash
# ReplicaSet과 ReplicaSet의 관리를 받던 pod 전부 삭제
kubectl delete rs <replicaset-name>

# ReplicaSet만 삭제되고 생성되어있던 Pod는 그대로 유지 (orphan=고아)
# ReplicaSet을 교체하고 싶을 때 사용
kubectl delete rs <replicaset-name> --cascade=orphan
```

<br>

Pod가 ReplicaSet에 의해 관리되는지 아닌지 확인하는 방법

``` bash
kubectl get pod <pod-name> -o jsonpath="{.metadata.ownerReferences[0].name}"
```

<br>

Gracefully하게 ReplicaSet과 Pod 삭제방법

``` bash
# Pod의 개수를 0으로 변경함으로써 기존에 생성되었던 Pod 전부 삭제
kubectl scale rs/<replicaset-name> --replicas 0

# ReplicaSet 삭제
kubectl delete rs/<replicaset-name>
```

<br>

**ReplicaSet Pod Template 변경시 주의사항**

-   기존에 ReplicaSet에 의해 생성된 Pod가 있을때 ReplicaSet의 Pod Template을 변경하더라도 기존의 Pod를 삭제하고 새로운 Pod를 생성하지 않는다.
-   ReplicaSet의 변경된 Pod Template을 적용시키려면 replicas 값이 변경되고나 기존의 Pod가 삭제되어 새로 Pod가 만들어 져야 하는 경우에 적용된다.

<br>

**Pod Template 이미지 변경을 통한 롤백 디버깅**

``` bash
# ReplicaSet 이미지를 1.0버전으로 변경
kubectl set image rs/<replicaset-name> <container-name>=<image>

# ReplicaSet과 Pod Template 확인 
kubectl get rs <replicaset-name> -o wide 

# Pod의 Owner 확인
kubectl get pod <pod-name> -o jsonpath="{.metadata.ownerReferneces[0].name}"

# 실행 중인 2.0 버전의 모든 Pod Label 변경
kubectl label pod <pod-name> app=to-be-fixed --overwrite
kubectl label pod <pod-name> app=to-be-fixed --overwrite
kubectl label pod <pod-name> app=to-be-fixed --overwrite

# 레이블을 변경한 Pod의 owner 확인
kubectl get pod <pod-name> -o jsonpath="{.metadata.ownerReferences[0].name}"

# 이제 ReplicaSet의 selector Label을 가지는 Pod가 없기때문에 replicas 만큼의 Pod가 자동생성됨
# myapp-replicaset-4zzpm   1/1     Running   0          21m   app=to-be-fixed
# myapp-replicaset-5g78p   1/1     Running   0          21m   app=to-be-fixed
# myapp-replicaset-bzx59   1/1     Running   0          24s   app=my-app
# myapp-replicaset-f25z7   1/1     Running   0          24s   app=my-app
# myapp-replicaset-j2tqx   1/1     Running   0          24s   app=my-app
# myapp-replicaset-tlfkc   1/1     Running   0          21m   app=to-be-fixed

# app=to-be-fixed가 된 Pod는 더이상 ReplicaSet의 관리를 받지 않기 때문에 Log를 확인해보며 디버깅을 자유롭게 할 수있게됨
kubectl logs myapp-replicaset-4zzpm
```

위와 같이 처리하게 되면 개발자는 문제가 어디서 생겼는지 디버깅 할 수 있는 시간을 벌 수 있으면서 사용자는 요청에 대한 정상적인 응답을 받을 수 있게 됨

<br>

etc

``` bash
kubectl get pod <pod-name>
# 특정 pod 조회

kubectl describe pod <pod-name>
# 특정 pod의 정보 및 이벤트 확인

kubectl get pod -o wide
# pod 목록을 보다 자세히 출력

kubectl get pod <pod-name> -o yaml
# <pod-name> 을 yaml 형식으로 출력

kubectl get pod -w
# kubectl get pod 의 결과를 계속 보여주며(watch mode), 변화가 있을 때만 업데이트

kubectl get pod <pod-name> -o jsonpath="{.status.podIP}"
# 특정 pod의 IP주소 가져오기
```

---

#### **3\. Deployment**

-   Deployment는 Pod와 Replicaset에 대한 관리를 제공하는 단위
-   여기서 관리의 의미는 Self-Healing, Scaling, Rollout과 같은 기능을 포함하며 쉽게말해 Pod를 한번 감싼 개념
-   Pod를 Deployment로 배포함으로써 여러 개로 복제된 Pod, 여러 버전의 Pod를 안전하게 관리 가능

<br>

**Deployment를 사용하는 이유**

-   만약 새로운 버전의 이미지를 적용해야 하는 상황이 왔을 때 ReplicaSet의 이미지를 업그레이드 하고 실행중인 Pod를 제거해야만 replicas에 의해 새 버전의 이미지가 적용된 Pod가 생성된다.
-   이때 새 버전에서 에러가 발생하여 다시 이미지를 다운그레이드 해야하는 상황이 오게되면 위와 같은 동작을 수작업으로 반복해야 한다.
-   여기서 우리는 Pod Template 이미지가 바뀔 때마다 쿠버네티스가 알아서 ReplicaSet을 생성하고 이전 Pod를 제거해주는 방법을 생각해볼 수 있다.
-   Deployment는 Pod 배포 자동화를 위한 오브젝트 (ReplicaSet + 배포전략)
-   Deployment 는 ReplicaSet을 여러개 가질 수 있으며 ReplicaSet의 replicas를 자동 조정하여 필요한 ReplicaSet을 쉽게 사용할 수 있다.

<br>

**Deployment 생성**

간단한 Deployment 예시

``` yaml
apiVersion: apps/v1 # kubernetes resource 의 API Version
kind: Deployment # kubernetes resource name
metadata: # 메타데이터 : name, namespace, labels, annotations 등을 포함
  name: nginx-deployment
  labels:
    app: nginx
spec: # 메인 파트 : resource 의 desired state 를 명시
  replicas: 3 # 동일한 template 의 pod 을 3 개 복제본으로 생성
  selector:
    matchLabels:
      app: nginx
  template: # Pod 의 template
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx # container 의 이름
        image: nginx:1.14.2 # container 의 image
        ports:
        - containerPort: 80 # container 의 내부 Port
```

<br>

**Deploymet 롤아웃**

전략1 - Recreate 배포

-   새로운 버전을 배포하기 전에 이전 버전이 즉시종료됨
-   컨테이너가 정상적으로 시작되기 전까지 서비스하지 못함
-   replicas 수만큼의 컴퓨팅 리소스 필요
-   서비스 운영단계에서는 부적합, 개발단계에서 유용

```  yaml
# Recreate 작성방법
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: Recreate
  template:
    metadata:
      name: my-app
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: <image>
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "64Mi"
            cpu: "50m"
```

<br>

전략2 - RollingUpdate 배포 

-   새로운 버전을 배포하면서 이전 버전을 종료
-   서비스 다운 타임 최소화
-   동시에 실행되는 Pod의 개수가 replicas를 넘게 되므로 컴퓨팅 리소스가 더 많이 필요

<br>

**RollingUpdate 속도 제어 옵션**

1\. maxUnavailable

-   롤링 업데이트를 수행하는 동안 유지하고자 하는 최소 Pod의 비율(수)을 지정할 수 있음
-   최소 Pod 유지 비율 = 100 - maxUnavailable 값
-   ex) replicas: 10, maxUnavailable: 30% 이면 즉시 3개의 Pod를 Scale Down하게 되고 그 이후부터는 새로운 버전의 Pod생성과 이전 버전의 Pod종료가 반복됨 -> 즉 롤아웃을 수행할 때 replicas수의 70% 이상의 Pod를 항상 Running 상태로 유지하겠다는 의미

<br>

2\. maxSurge

-   롤링 업데이트를 수행하는 동안 허용할 수 있는 최대 Pod의 비율(수)을 지정할 수 있음
-   최대 Pod 허용비율 = maxSurge 값
-   ex) replicas: 10, maxSurge: 30% 이면 새로운 버전의 Pod를 3개까지 즉시 생성할 수 있고 Pod 생성과 이전 버전의 Pod 종료를 진행하면서 총 Pod의 수가 replicas 수의 130%를 넘지 않도록 유지해야 함

<br>

3\. maxUnavailable, maxSurge가 필요한 이유

-   모든 Old Pod를 New Pod로 전환하는데 시간을 최소화 할 수 있다.
-   새로운 Pod를 replicas 수만큼 미리 배포한다면 리소스가 부족할 수 있다.
-   maxUnavailable을 이용해서 최소 서비스 운영에 영향을 주지 않을 만큼 유지해야 하는 Pod수를 선언할 수 있다.
-   maxSurge로 어떤 시점에 동시에 존재할 수 있는 최대 Pod 수를 선언하여 배포 속도를 조절함과 동시에 리소스를 제어할 수 있다.
-   즉 유지해야할 Pod 수의 상한선과 하한선을 쿠버네티스에 알리기 위한 옵션

``` yaml
#RollingUpdate 작성방법
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  replicas: 5
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 2
      maxSurge: 1
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: <image>
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "64Mi"
            cpu: "50m"
```

<br>

Deployment Pod replicas 변경

``` bash
# deployment 배포 진행중/완료 상태 확인
kubectl rollout status deployment/<deployment-name>

# deployment의 replicas 변경
kubectl scale deployment <deployment-name> --replicas=5
```

<br>

Deployment Pod Template 이미지 변경

-   Pod Template이 변경되기 때문에 Deployment가 새로운 해시값을 가지는 ReplicaSet을 생성한다.
-   이전 ReplicaSet은 자신이 관리하는 Pod를 모두 제거하고 새로운 ReplicaSet은 새로운 Pod를 replicas 수만큼 생성한다.
-   이미지 변경이 아닌 레이블 변경이 이루어져도 해시값이 달라지기 때문에 위와 동일한 동작을 함

``` bash
# deployment 이미지 변경 (뒤에 --record를 적으면 history에 실행한 명령어가 적힘)
kubectl set image deployment/<deployment-name> <container-name>=<image>

# ReplicaSet 확인
kubectl get rs <resplicaset-name>
# NAME                DESIRED   CURRENT   READY   AGE
# my-app-569b7cb744   0         0         0       23m
# my-app-6dfd664d67   3         3         3       11m

# Deployment 이벤트 확인
kubectl describe deployment <deployment-name>
#  Normal  ScalingReplicaSet  26m   deployment-controller  Scaled up replica set my-app-569b7cb744 to 3
#  Normal  ScalingReplicaSet  13m   deployment-controller  Scaled up replica set my-app-6dfd664d67 to 1
#  Normal  ScalingReplicaSet  13m   deployment-controller  Scaled down replica set my-app-569b7cb744 to 2
#  Normal  ScalingReplicaSet  13m   deployment-controller  Scaled up replica set my-app-6dfd664d67 to 2
#  Normal  ScalingReplicaSet  13m   deployment-controller  Scaled down replica set my-app-569b7cb744 to 1
#  Normal  ScalingReplicaSet  13m   deployment-controller  Scaled up replica set my-app-6dfd664d67 to 3
#  Normal  ScalingReplicaSet  13m   deployment-controller  Scaled down replica set my-app-569b7cb744 to 0
```

<br>

**Deployment 롤백**

Deployment는 롤아웃 히스토리를 Revision으로 관리

``` bash
# Revision 목록 조회 - 간단한 배포 기록 확인
kubectl rollout history deployment/<deployment-name>

# Revision 상세 조회 - 배포된 Pod Template 확인
kubectl rollout history deployment/<deployment-name> --revision=<revision-number>

# 이전 혹은 특정 Revision으로 롤백
kubectl rollout undo deployment/<deployment-name>
kubectl rollout undo deployment/<deployment-name> --to-revision=<revision-number>

# 롤백 사유 남기기
kubectl annotate deployment/<deployment-name> kubernetes.io/change-cause="<message>"
```

``` yaml
# annotation 작성방법
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
  annotation:
    kubernetes.io/change-cause: "initial image 1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: my-app
        project: fastcampus
        env: production
    spec:
      containers:
      - name: my-app
        image: <image>
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "64Mi"
            cpu: "50m"
```

---

#### **4\. Service**

-   Service는 쿠버네티스에 배포한 애플리케이션(pod)을 외부에서 접근하기 쉽게 추상화한 리소스이다.
    -   [https://kubernetes.io/ko/docs/concepts/services-networking/service/](https://kubernetes.io/ko/docs/concepts/services-networking/service/)
-   pod는 항상 고유 IP를 할당받고 생성되지만 언제든지 죽었다가 살아날 수 있으며 그 과정에서 IP는 항상 재할당 받기에 클러스터 외부에서 고정된 IP로 원하는 Pod에 접근할 수 없다.
-   따라서 클러스터 외부에서 Pod에 접근할 때는, Pod의 IP가 아닌 Service를 통해서 접근하는 방식을 거친다.
-   Service는 고정된 IP를 가지며, Service는 하나 혹은 여러개의 Pod와 매칭될 수 있다.
-   따라서 클라이언트가 Service의 단일 endpoint 주소로 접근하면, 실제로 Service에 매칭된 Pod에 접속할 수 있게 된다.

<br>

**Service의 Type**

-   **NodePort** 라는 type을 사용했기 때문에, minikubefksms kubernetes cluster 내부에 배포된 서비스에 클러스터 외부에서 접속할 수 있었다.
    -   이때 IP는 노드의 IP주소, Port는 서비스에 할당받은 Port를 사용
-   **LoadBalancer** 라는 type을 사용해도 마찬가지로 클러스터 외부에서 접속할 수 있지만, LoadBalancer를 사용하기 위해서는 LoadBalancing 역할을 하는 모듈이 추가적으로 필요하다.
-   **ClusterIP**라는 type은 고정된 IP, Port를 제공하지만, 클러스터 내부에서만 접근할 수 있는 대역의 주소가 할당된다.
    -   개발단계에서 port-forward를 통해 localhost로 요청이 정상적으로 작동하는지 확인할 수 있음
-   실무에서는 주로 kubernetes cluster에 MetalLB와 같은 LoadBalancing 역할을 하는 모듈을 설치한 후, LoadBalancer type으로 서비스를 expose하는 방식을 사용한다.
    -   NodePort는 Pod가 어떤 Node에 스케줄링될 지 모르는 상황에서, Pod가 할당된 후 해당 Node의 IP를 알아야 한다는 단점이 존재한다.

<br>

**기본 Service 관련 kubectl 명령어**

``` bash
# 네임스페이스 생성
kubectl create namespace <namespace-name>

# 네임스페이스의 모든 리소스 조회
kubectl get all -n <namespace-name>

# 네임스페이스의 Service 상세 조회
kubectl get svc <service-name> -o wide -n <namespace-name>

# Service ClusterIP 조회
kubectl get svc <service-name> -o jsonpath="{.spec.clusterIP}" -n <namespace-name>

# 네임스페이스의 모든 Endpoint 리소스 확인
kubectl get endpoints -n <namespace-name>
```

<br>

**ClusterIP 서비스로 Pod 노출하는 방법**

**1\. 환경변수 이용**

-   서비스가 먼저 배포된 이후에 pod가 배포되기 때문에 쿠버네티스는 각 pod가 배포될때 이미 알고있는 정보인 SERVICE\_HOST와 SERVICE\_PORT를 각 컨테이너의 환경변수로 전달시킨다.
-   그렇기 때문에 역으로 Pod의 환경변수로 Service를 추적할 수 있다.
-   다른말로 pod가 생성된 이후 나중에 생성된 서비스에 대한 환경변수는 알 수 없다는 것을 유의

``` bash
# 특정 컨테이너 환경변수 확인 (-c 옵션으로 특정 컨테이너 지정하지 않으면 첫번째 컨테이너가 default)
kubectl exec <pod-name> -n <namespace-name> -- env | grep <upper-service-name>

# PAYMENT_PORT_80_TCP_PROTO=tcp
# PAYMENT_PORT_80_TCP=tcp://10.98.54.158:80
# PAYMENT_SERVICE_HOST=10.98.54.158
# PAYMENT_PORT=tcp://10.98.54.158:80
# PAYMENT_SERVICE_PORT=80
# PAYMENT_PORT_80_TCP_ADDR=10.98.54.158
# PAYMENT_PORT_80_TCP_PORT=80
```

<br>

``` bash
# 네임스페이스의 order Pod의 컨테이너 쉘 접속
kubectl exec -it <order-pod> -n <namespace-name> -- sh

# payment 서비스 환경변수를 이용하여 Payment 호출
curl $PAYMENT_SERVICE_HOST:$PAYMENT_SERVICE_PORT
```

-   위와같이 특정 pod의 컨테이너에 접속해보면 pod가 소속된 서비스 뿐만아니라 pod가 생성되기 전 배포된 현재 네임스페이스의 모든 서비스에 대한 환경변수를 컨테이너가 가지고 있기 때문에 컨테이너 내부에서 다른 pod로 요청을 보낼 수 있다.

<br>

**2\. 서비스 이름 이용 - 쿠버네티스 DNS 서버**

``` bash
# 서비스 이름으로 다른 pod 요청/응답확인
kubectl exec -it <order-pod> -n <namespace-name> -- curl -s <service-name>:80

# 네임스페이스에 있는 특정 pod 컨테이너의 /etc/hosts 확인
kubectl exec -it <pod-name> -n <namespace-name> -- cat /etc/hosts

# 127.0.0.1       localhost
# 192.168.27.174  order-5d45bf5796-zvhrj # 자기 자신의 호스트와 IP만 있음
```

-   hosts 파일에 정의되지 않은 다른 서비스의 도메인 이름으로 요청이 가능한 이유? -> kube-system 네임스페이스에서 실행중인 kube-dns pod를 통해 도메인 네임을 찾게 됨

<br>

```bash
# kube-system 네임스페이스의 모든 kube-dns 리소스 조회
kubectl get all -n kube-system | grep kube-dns

# service/kube-dns  ClusterIP  10.96.0.10
```

```bash
# 네임스페이스의 order Pod의 컨테이너 쉘 접속
kubectl exec -it <order-pod> -n <namespace-name> -- sh

# DNS 서버 설정 확인 
cat /etc/resolv.conf

# nameserver 10.96.0.10
# search <namespace-name>.svc.cluster.local svc.cluster.local cluster.local
```

-   nameserver: 컨테이너에서 사용할 DNS 서버 주소
-   search: 클러스터 내에서 사용할 도메인 접미사 정의
-   svc.cluster.local: 모든 클러스터 로컬 서비스 이름에 사용되는 도메인 접미사
-   FQDN(fully qualified domain name): <서비스이름>.<네임스페이스>.svc.cluster.local
-   FQDN을 이용해서 DNS 쿼리를 실행(DNS Server에서 Service IP를 조회)

<br>

``` bash
# 셋 다 동일한 결과
# FQDN으로 특정 서비스 호출 
curl <service-name>.<namespace-name>.svc.cluster.local

# 클러스터 접미사(svc.cluster.local)를 제거하고 Payment 서비스 호출
curl <service-name>.<namespace-name>

# 서비스 이름만으로 Payment 서비스 호출 (namespace-name을 생략하면 서비스 매니패스트에 작성된 namespace를 default로 가짐을 주의)
curl <service-name>
```

<br>

**3\. 서비스 이름으로 다른 네임스페이스에 있는 서비스 호출**

``` bash
# 특정 레이블을 가진 클러스터의 모든 리소스 조회
kubectl get all -l <label-key>=<label-value> --all-namespaces

# 특정 레이블을 가진 서비스 엔드포인트 조회
kubectl get endpoints -l <label-key>=<label-value> --all-namespaces

# 특정 네임스페이스의 컨테이너에서 다른 네임스페이스에 있는 서비스 CluserterIP로 요청 실행 후 응답 확인 
kubectl exec <pod-name> -n <namespace-name> -- curl -s <other-namespace-cluster-ip>

# 특정 네임스페이스의 컨테이너에서 다른 네임스페이스에 있는 서비스 도메인 이름으로 요청 실행 후 응답 확인 
kubectl exec <order-pod> -n <namespace-name> -- curl -s <other-service-name>.<other-namespace-name>
```

<br>

**Service ClusterIP 특징**

-   Service는 파드 집합에 대한 단일 엔드포인트를 생성한다.
-   Service를 생성하면 ClusterIP가 생성된다.
-   ClusterIP는 클러스터 내부에서만 접속할 수 있다.

<br>

**ClusterIP를 이용해서 다른 Pod에게 요청을 보내는 방법**

-   특정 애플리케이션 파드를 위해 배포된 Service 이름을 알아낸다.
-   애플리케이션 컨테이너에서 OOO\_SERVICE\_HOST 환경변수로 Service IP를 알아낼 수 있다.
-   단, Pod보다 늦게 생성한 Service 환경변수는 사용할 수 없다. + 다른 네임스페이스의 Service는 환경변수로 설정되지 않는다.
-   애플리케이션 컨테이너에서 Service IP대신 Service 이름을 도메인으로 요청을 보낼 수 있다.
-   애플리케이션 컨테이너에서 Service Port는 OOO\_SERVICE\_PORT 환경변수를 이용한다.

<br>

**Service NodePort 특징**

-   클러스터 내 모든 노드에 포트 할당은 Service를 NodePort 타입으로 생성 했을 때 일어난다.
-   노드의 External IP와 서비스 NodePort를 이용해서 pod에 접근 할 수 있다.
-   서비스 ClusterIP도 여전히 클러스터 내부에서 사용할 수 있다.

<br>

**NodePort를 이용해서 다른 Pod에게 요청을 보내는 방법**

-   GKE사용자라면 NodePort에 대한 인바운드 트래픽 허용 정책을 클라우드 서비스에 설정한다.
-   노드 IP와 NodePort를 이용해서 원하는 파드 집합에 요청을 실행한다.

<br>

**Service LoadBalancer 특징**

-   LoadBalancer 타입의 서비스를 생성하면 클라우드 서비스의 로드밸런서가 실행한다.(on-premise는 별도 Load Balancer 설치 필요)
-   로드밸런서의 IP가 Service의 External IP로 할당된다.
-   Service의 External IP이자 로드밸런서 IP로 외부에서 파드에 접근할 수 있다.
-   서비스 ClusterIP, NodePort의 기능도 여전히 사용할 수 있다.

<br>

**LoadBalancer를 이용해서 다른 Pod에게 요청을 보내는 방법**

-   서비스의 External IP를 이용해서 원하는 파드 집합에 요청을 실행한다.

---

**Ingress & IngressController**

**Ingress가 왜 필요할까?**

-   서비스를 외부로 노출시키는 방법들은 external ip를 할당받고 그것을 통해 외부 트래픽을 받을 수 있다. 하지만 클러스터에 수많은 서비스가  존재하게 되면 클라이언트가 External IP를 관리하고 기억해야 하는 부담이 커지게 된다. 
-   이에따라 많은 서비스를 어떻게하면 단일 엔드포인트로 제공할 수 있을까에 대한 고민이 있을 수 있다.

<br>

**Ingress: Service 추상화, 의미있는 단일 엔드포인트 제공**

-   트래픽을 Service로 분산하기 위한 라우팅 규칙 모음
-   클라이언트가 호출한 Host 헤더나 path를 통해 Service를 구분하고 트래픽을 포워딩(ex. order.snackbar.com)
-   클라이언트가 하나의 ip로 접근할 수 있게 도와줌

<br>

**Ingress Controller: Ingress 규칙에 따라 트래픽 분산을 실행하기 위한 리소스**

-   쿠버네티스 클러스터 제공자가 구현한 Ingress Controller마다 기능이 다르다.(AWS, GCP, nginx ingress controller)
-   쿠버네티스 지원 Ingress Controller: [https://bit.ly/3GkpoZq](https://bit.ly/3GkpoZq)

<br>

**Ingress 생성 방법**

**\- multiple host**

``` yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: <name>
  namespace: <namespace>
  labels:
    <key>: <value>
spec:
  defaultBackend:  # 매치되는 라우트가 없을 때 home 서비스의 포트 80으로 포워딩
    service:
      name: home
      port:
        number: 80
  rules:  # host들의 목록을 선언, http.paths: 서비스 백엔드로 매핑할 path 목록
  - host: order.fast-snackbar.com  # Host: order.fast-snackbar.com 매치
  	http:
      paths:
      - pathType: Prefix 
        path: /  # /로 시작하는 모든 경로와 매치
        backend:  # 두 조건(host, path)을 만족하면 order 서비스의 포트 80으로 연결
          service:
            name: order
            port:
              number: 80
```

-   여기서 host는 가짜 도메인이기 때문에 /etc/hosts 파일을 ingress의 IP주소와 매핑되도록 변경해주어야 한다.

<br>

**\- Single host**

``` yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: <name>
  namespace: <namespace>
  labels:
    <key>: <value>
spec:
  defaultBackend:  # 매치되는 라우트가 없을 때 home 서비스의 포트 80으로 포워딩
    service:
      name: home
      port:
        number: 80
  rules:  # host들의 목록을 선언, http.paths: 서비스 백엔드로 매핑할 path 목록
  - http:  # host를 선언하지 않았으므로 Host 헤더를 확인하지 않음
      paths:
      - pathType: Prefix 
        path: /order  # /order 로 시작하는 모든 경로와 매치
        backend:  # 두 조건(host, path)을 만족하면 order 서비스의 포트 80으로 연결
          service:
            name: order
            port:
              number: 80
```

<br>

**\- Kubectl 명령어**

``` bash
# 네임스페이스에 인그레스 조회
kubectl get ingress <ingress-name> -n <namespace-name>

# 네임스페이스에 인그레스 로드밸런서 IP 조회
kubectl get ingress <ingress-name> -n <namespace-name> -o jsonpath="{.status.loadBalancer.ingress[0].ip}"

# 호스트 헤더를 통한 request
curl -H "Host: <ingress 호스트 목록>" --request GET <INGRESS_IP>

# path를 통한 request
curl --request GET <INGRESS_IP>/<path>
```

---

**Pod livenessProbe(직역: 생사확인)**

-   컨테이너의 health 상태를 확인하는 쿠버네티스 프로세스는 워커노드에서 실행되고 있는 kubelet 이다.
-   pod가 정상적으로 배포되더라도 내부의 애플리케이션 컨테이너가 요청을 처리할 수 없다면 아무런 의미 x
-   쿠버네티스가 컨테이너 상태를 확인할 수 있도록 엔드포인트를 livenessProbe로 정의해서 쿠버네티스에 알려주게되고 kubelet은 livenessProbe 엔드포인트를 이용해서 컨테이너의 상태를 체크할 수 있게됨
-   libenessProbe 엔드포인트를 주기적으로 실행 -> 적절한 응답을 받지 못했을때 자체적으로 컨테이너를 재시작하는 매커니즘

<br>

**Pod livenessProbe 선언방법**

HttpGet livenessProbe 선언 - HTTP status code로 살아있는지 확인하는 방법

``` yaml
apiVersion: v1
kind: Pod
metadata:
  name: <pod-name>
spec:
  containers:
  - name: <container-name>
    image: <image-path>
    ports:
      - containerPort: 8080
    livenessProbe:
      httpGet:  # probe 엔드포인트
        path: /
        port: 8080
      initialDelaySeconds: 30  # 컨테이너 시작 후 몇 초후에 probe를 시작할 것인가
      periodSeconds: 5  # probe 실행주기
      successThreshold: 1  # 몇개 성공 시 실패 횟수를 초기화 할 것인가
      failureThreshold: 1  # 연속으로 몇 번 실패 했을 때 컨테이너를 재시작할 것인가
      timeoutSeconds: 10  # 응답을 몇 초 만에 받아야 하는가
```

<br>

**Pod readinessProbe(직역: 준비성확인)**

-   데이터 로드 등과 같은 이유로 아직 준비가 안된 파드를 제외하여 클라이언트에게 불편한 경험을 주지 않을 수 있음
-   쿠버네티스가 컨테이너 준비 정도를 확인할 수 있도록 Pod readinessProbe를 선언
-   일정 수준 이상 연속해서 실패하면 서비스 엔드포인트에서 파드를 제거

<br>

**Pod readinessProbe 선언방법**

Exec readinessProbe 선언 - process exit status code로 준비 상태를 확인하는 방법

``` yaml
apiVersion: v1
kind: Pod
metadata:
  name: <pod-name>
spec:
  containers:
  - name: <container-name>
    image: <image-path>
    ports:
      - containerPort: 8080
    readinessProbe:
      exec:  # 컨테이너에서 실행할 명령어 probe
        command:
        - ls
        - /var/ready  # /var/ready라는 폴더가 존재하면 정상적으로 작동한다는 의미
      initialDelaySeconds: 30  # 컨테이너 시작 후 몇 초후에 probe를 시작할 것인가
      periodSeconds: 5  # probe 실행 주기
      successThreshold: 1  # 몇 개 성공시 실패 횟수를 초기화 할 것인가
      failureThreshold: 1  # 연속으로 몇번 실패 했을 때 파드가 준비되지 않았다고 표시할 것인가
      timeoutSeconds: 10  # 응답을 몇초만에 받아야 하는가
```

---

**ConfigMap**

\- 쿠버네티스는 Pod로부터 설정파일을 분리해서 관리할 수 있는 방법을 제공

\- ConfigMap 오브젝트로 설정파일을 관리하고 Pod와 분리할 수 있다.

<br>

**Pod로부터 어플리케이션 설정 정보 분리하기**

-   Pod의 컨테이너 환경변수가 ConfigMap의 값을 참조 -> ConfigMap을 먼저 선언해야 함
-   Pod 볼륨으로 ConfigMap을 사용

<br>

**ConfigMap을 사용하는 장점**

-   Pod가 종료되고 다시 생성되더라도 동일한 Pod Manifest에는 동일한 ConfigMap이름으로 참조하기 때문에 설정 파일의 정보를 재사용 할 수 있다.
-   ConfigMap 이름으로 설정값들을 참조하기 때문에 설정값의 변경이 자유롭다. -> Pod와의 의존성이 적어지고 설정파일을 독립적으로 관리

<br>

**ConfigMap 생성 방법**

1\. 리터럴 방식

key=value 를 직접 커맨드 라인에 작성하는 방법

kubectl create configmap <name> --from-literal=key=value

Example

``` bash
kubectl create configmap greeting-config --from-literal=STUDENT_NAME=홍은표 --from-literal=MESSAGE=안녕
```

``` yaml
apiVersion: v1
kind: Pod
metadata:
  name: <pod-name>
  labels:
    <key>: <value>
spec:
  containers:
  - name: <container-name>
    image: <image-path>
    ports:
    - containerPort: 8080
    env:
    - name: STUDENT_NAME
      valueFrom:
        configMapKeyRef:
          key: STUDENT_NAME
          name: greeting-config
    - name: MESSAGE
      valueFrom:
        configMapKeyRef:
          key: MESSAGE
          name: greeting-config
    - name: GREETING 
      value: $(MESSAGE) $(STUDENT_NAME)
```

-   모든 key/value 쌍을 선언해야 할때 위와같은 방식으로 작성하면 yaml 파일이 길어지기 때문에 greeting-map에 선언한 모든 key/value 쌍을 envFrom으로 컨테이너 환경변수로 설정할 수 있다.

<br>

``` yaml
apiVersion: v1
kind: Pod
metadata:
  name: <pod-name>
  labels:
    <key>: <value>
spec:
  containers:
  - name: <container-name>
    image: <image-path>
    ports:
    - containerPort: 8080
    envFrom:
      - configMapRef:
          name: greeting-config
    env:
    - name: GREETING
      value: $(MESSAGE)! $(STUDENT_NAME)
```

<br>

2\. 파일 및 폴더를 지정하는 방식

kubectl create configmap <name> --from-file=파일이나 디렉토리 경로

\- 파일 이름: key, 파일 내용: value

Example
![](/images/../images/2023-03-12-01-04-49.png){: .align-center}

``` bash
kubectl create configmap greeting-config-from-file --from-file=configs

kubectl get configmap greeting-config-from-file -o yaml
```

``` yaml
apiVersion: v1
data:
  MESSAGE: 안녕하세요
  STUDENT_NAME: 홍은표
kind: ConfigMap
metadata:
  creationTimestamp: "2022-04-04T01:59:08Z"
  name: greeting-config-from-file
  namespace: snackbar
  resourceVersion: "43931"
  uid: 337df33c-8db6-444f-a01a-52e34d6c5d4d

```

-   이후 Pod Manifest에서 envFrom으로 1번 리터럴 방식과 동일하게 작성

<br>

3\. Pod 볼륭을 이용한 ConfigMap 사용

ConfigMap을 생성 -> ConfigMap 타입의 Volume을 Pod에서 선언 -> 컨테이너가 해당 볼륨을 마운트

Example

``` yaml
spec:
  volumes: # Pod에서 사용할 볼륨 목록 선언 
  - name: app-config  # 컨테이너에서 참조할 볼륨 이름
    configMap:
      name: nginx-config  # 참조할 ConfigMap 이름
  containers:
  - name: nginx
    image: nginx
    ports:
      - containerPort: 80
    volumeMounts: # 컨테이너에서 Pod 볼륨 마운트 선언
    - name: app-config  # 마운팅할 Pod 볼륨 이름
      mountPath: /etc/nginx/conf.d  # 컨테이너 안에서 마운팅할 경로
```

``` bash
# 실제로 컨테이너에 config 파일이 마운트 되었는지 확인
kubectl exec web-server -c nginx -- cat /etc/nginx/conf.d/server.conf
```

---

**Secret**

-   애플리케이션 설정파일에는 서버 접속을 위한 비밀번호, 암호화를 위한 public/private key등 노출이 되면 안되는 민감 정보도 있음
-   이러한 민감 정보를 관리하기 위한 쿠버네티스 오브젝트가 Secret
-   ConfigMap처럼 민감한 데이터를 key/value 쌍으로 저장
-   쿠버네티스가 Secret 값을 Base64로 인코딩해서 관리(보안)
-   컨테이너에서 Secret 값을 읽을 때는 디코딩되어 전달
-   Pod 선언 시 Secret 볼륨이나 환경변수를 통해서 Secret 값을 사용 가능

\-> 쿠버네티스에서 보안관리가 가능하기 때문에 애플리케이션의 민감 데이터를 관리하기 위해 별도의 서버를 실행할 필요가 없음

<br>

**Secret 사용 방법(ConfigMap과 유사하게 사용)**

1.  컨테이너 env.valueFrom.secretMapKeyRef 사용
2.  컨테이너 envFrom.secretRef 사용
3.  Secret을 Pod 볼륨으로 연결하고 컨테이너에서 마운트

Example) TLS인증서를 Secret 볼륨으로 관리하기

![](/images/../images/2023-03-12-01-05-03.png){: .align-center}

``` bash
# Secret 타입 - 기본 generic or TLS 클라이언트나 서버를 위한 데이터 kubernetes.io/tls
kubectl create secret generic tls-config --from-file=secrets/https.cert --from-file=secrets/https.key

# pod를 배포한 이후
# curl 클라이언트가 nginx 서버로부터 받은 인증서를 신뢰할 수 있도록 자체 서명한 인증서(secrets/https.cert)를 서버 인증서 검증에 사용 설정
curl --cacert secrets/https.cert -sv https://<domain-name>:<port>/<endpoint>
```

---

#### **5\. PVC**

-   PVC(Persistent Volume Claim)는 PV(Persistent Volume)이라는 리소스가 함께 다닌다. 이 리소스들은 stateless 한 Pod에 영구적으로 데이터를 보존하고 싶은 경우 사용하는 리소스이다. (도커의 docker run -v 옵션과 유사한 역할을 함)
-   PV는 관리자가 생성한 실제 저장 공간의 정보를 담고 있고, PVC는 사용자가 요청한 저장 공간의 스펙에 대한 정보를 담고 있는 리소스이다.
-   PVC를 사용하면 여러 pod간의 data 공유도 쉽게 가능하다.
-   Pod 내부에서 작성한 데이터는 기본적으로 언제든지 사라질 수 있기에, 보존하고 싶은 데이터가 있다면 Pod에 PVC를 mount해서 사용해야 한다는것을 기억하자
-   PVC는 Namespace Object이기 때문에 Namespace에 디펜던시가 걸리지만 PV는 Cluster Role과 비슷하게 클러스터에서 공용으로 사용할수 있는 객체라는것을 명심
-   PV, PVC, Storageclass에 대한 자세한 설명 [여기](https://do-hansung.tistory.com/57 "https://do-hansung.tistory.com/57") 참고

<br>

**PVC 생성**

-   minikube를 생성하면 기보적으로 minikube와 함께 설치되는 storageclass가 존재한다.
-   kubectl get storageclass를 통해 확인 가능하며 storageclass는 PVC를 생성하면 PVC의 스펙에 맞는 PV를 그 즉시 자동생성해준뒤, PVC와 매칭시켜준다고 이해하면 된다.(dynamic provisioning을 지원)

![](/images/../images/2023-03-12-01-05-16.png){: .align-center height="70%" width="70%"}

``` yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: myclaim
spec:
  accessModes:
    - ReadWriteMany # ReadWriteOnce, ReadWriteMany 옵션을 선택할 수 있음(여러개 파드, 하나의 파드)
  volumeMode: Filesystem
  resources:
    requests:
      storage: 10Mi # storage 용량 설정
  storageClassName: standard # 방금 전에 확인한 storageclass 의 name 을 입력
```

``` bash
vi pvc.yaml

kubectl apply -f pvc.yaml

kubectl get pvc,pv
```

![](/images/../images/2023-03-12-01-05-27.png){: .align-center}

-   pvc와 함께 pv가 생성된 것을 확인할 수 있다.(AGE 6s 확인)

<br>

**Pod에서 PVC사용**

-   volumeMounts, volumes 부분이 추가됨

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mypod
spec:
  containers:
    - name: myfrontend
      image: nginx
      volumeMounts:
      - mountPath: "/var/www/html" # mount 할 pvc 를 mount 할 pod 의 경로를 적는다
        name: mypd # 어떤 이름이든 상관없으나, 아래 volumes[0].name 과 일치해야 한다
  volumes:
    - name: mypd # 어떤 이름이든 상관없으나, 위의 volumeMounts[0].name 과 일치해야 한다
      persistentVolumeClaim:
        claimName: myclaim # mount 할 pvc 의 name을 적음 (방금 생성한 pvc)
```

``` bash
vi pod-pvc.yaml

kubectl apply -f pod-pvc.yaml
```

<br>

pod에 접속하여 mount한 경로와 그 외의 경로에 파일을 생성

``` bash
kubectl exec -it mypod -- bash
# pod 접속

touch hello-world
# 파일생성

cd /var/www/html

touch hello-world
```

pod 삭제 후 pvc와 pv가 그대로 남아있는지 확인

<br>

```  bash
kubectl delete pod mypod
# pod 삭제

kubectl get pvc,pv
# 그대로 남아있음을 확인
```

<br>

 해당 pvc를 mount하는 pod를 다시 생성한 후 접속하여 아까 작성한 hello-world가 남아있는지 확인

``` bash
kubectl apply -f pod-pvc.yaml

kubectl exec -it mypod -- bash

ls
# hello-world 파일이 사라진 것을 확인할 수 있다.

cd /var/www/html

ls
# hello-world 파일이 보존된 것을 확인할 수 있다.
```

<br>

#### **6\. Node** 

**Master Node**

-   Cluster 관리를 위한 명령을 내리고 총괄하는 역할
-   Worker Node를 관리하고, Container 생성 요청에 따라 어떤 Worker Node에 Container를 띄울지 결정

<br>

**Worker Node**

-   마스터 노드의 명령에 따라 Container를 생성하는 노드
-   해당 노드 위에서 실행되고 있는 Container들의 상태를 주기적으로 마스터 노드에 보고

---

Reference

Doc(KOR) - [https://kubernetes.io/ko/docs/tutorials/kubernetes-basics/](https://kubernetes.io/ko/docs/tutorials/kubernetes-basics/)