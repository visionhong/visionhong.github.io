---
title:  "DVC(Data Version Control) with Docker"
folder: "tools"
categories:
  - tools
toc: true
tags:
  - DVC
  - Docker
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
header:
  teaser: "/images/2023-03-11-19-11-33.png"
---

![](/images/../images/2023-03-11-19-11-33.png){: .align-center height="30%" width="30%"}

DVC는 Data Version Control의 약어로 머신러닝 프로젝트의 Open-source Version Control System이다. DVC는 터미널에서 명령어로 동작하며 Git과 명령어가 굉장히 유사하기 때문에 빠르게 DVC를 습득할 수 있다.

DVC는  데이터 버전관리 외에도 MLOps의 구성요소인 ML experiment management, Deployment & Collaboration 기능을 제공하지만 이번 포스팅에서는 데이터를 원격저장소에 저장하고 다운받고 깃과 함께 버전관리 하는 것을 집중해서 다루려고한다.

또한 Dockerfile을 DVC를 활용하여 이미지로 build 해볼 것이다.

---

#### **1\. Install**

git 설치: [https://git-scm.com/downloads](https://git-scm.com/downloads)

dvc 설치

\- dvc 2.6.4 버전 다운로드
\- dvc\[all\]: dvc 의 remote storage 로 s3, gs, azure, oss, ssh 모두를 사용할 수 있도록 관련 패키지를 함께 설치하는 옵션

``` python
pip install 'dvc[all]' == 2.6.4

dvc --version
# 2.6.4
```

#### **2\. DVC Directory Setting**

1) 새 Directory 생성

```python
mkdir dvc-tutorial

cd dvc-tutorial
```

<br>

2) 해당 Directory를 git 저장소로 초기화

```python
git init
```

<br>

3) 해당 Directory를 dvc 저장소로 초기화

\- 폴더 내에 .git 폴더와 .dvc 폴더가 생성된 것 확인

``` python
dvc init

ls -a
```

![](/images/../images/2023-03-11-19-12-04.png){: .align-center height="70%" width="70%"}

<br>

#### **3\. DVC 기본 명령어 1**

1) dvc로 버전 tracking 할 data 생성

\- data라는 폴더를 생성하고 그 안에 이미지 4장을 생성하였다.

![](/images/../images/2023-03-11-19-12-13.png){: .align-center}

<br>

2) 생성한 데이터를 dvc로 tracking

\- data.dvc라는 파일이 생성되며 이 파일의 안에있는 metedata를 통해 원격 저장소와 연결이 된다.

```python
dvc add data

git add .gitignore data.dvc
```

<br>

3) dvc add에 의해 자동 생성된 파일 확인

```python
cat data.dvc
```

![](/images/../images/2023-03-11-19-12-26.png){: .align-center height="70%" width="70%"}

<br>

4) git commit 수행

```python
git commit -m add data.dvc
```

<br>

5) data가 실제로 저장될 remote storage를 세팅

\- 여기서 s3, gs, azure, oss, ssh, google drive 등 자신이 원하는 원격저장소를 선택할 수 있다.
\- 필자는 gpu서버의 공간을 원격 저장소로 지정하였다.(ssh)
\- dvc의 default remote storage로 gpu서버 내에 원격저장소로 활용할 폴더를 세팅
\- ssh는 password 혹은 ssh key를 통해 원격 저장소와 연동을 할 수가 있는데 공식 문서에서 ssh key를 추천하고있다.
\- private key 생성방법: [https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2)

```python
dvc remote add -d {storagename} ssh://{your/ip/path}/{your/dir/path}

dvc remote modify {storagename} user {user name}

dvc remote modify {storagename} port {port number}

dvc remote modify {storagename} keyfile {private key}
# private key: 원격 저장소 연결을 위한 ssh private key 파일 경로 작성
```

<br>

6) remote add로 자동생성된 .dvc/config 파일 확인

```python
cat .dvc/config
```

![](/images/../images/2023-03-11-19-12-37.png){: .align-center height="70%" width="70%"}

<br>

7) dvc config를 git commit

```python
git add .dvc/config

git commit -m "add remote storage"
```

<br>

8) dvc push

\- 데이터를 원격 저장소에 업로드한다.

\- 전송이 완료된 원격 저장소를 보면 알수없는 이름의 폴더와 파일이 생기게 된다.

```python
dvc push
```

<br>

#### **4\. DVC 기본 명령어 2**

1) dvc pull

\- 데이터를 삭제한 후 원격 저장소로부터 데이터를 다운로드 받는다.

```python
# dvc 캐시를 삭제합니다.
rm -rf .dvc/cache/
# dvc push 했던 데이터를 삭제합니다.
rm -rf data

dvc pull

ls data
```

![](/images/../images/2023-03-11-19-12-48.png){: .align-center height="70%" width="70%"}

<br>

2) dvc checkout

\- checkout은 데이터의 버전을 변경하는 명령어
\- 버전 번경 테스트를 위해, 새로운 버전의 data를 dvc push

```python
# new 라는 이름의 이미지 추가
dvc status
```

![](/images/../images/2023-03-11-19-12-55.png){: .align-center height="70%" width="70%"}

<br>

```python
dvc add data

git add data.dvc

git commit -m 'update data'

dvc push
# 원격 저장소에 새로운 파일이 등록되었는지 확인
```

\- 이제 다시 이전 버전으로 돌아가보자
\- 5개였던 파일이 4개로 돌아가게 된다.

<br>

```python
# git의 commit 정보 확인
git log --oneline

# data.dvc 파일을 이전 commit 버전으로 되돌림
git checkout <COMMIT_HASH> data.dvc

# data.dvc 의 내용을 보고 data 폴더를 이전 commit 버전으로 변경
dvc checkout

# 데이터가 변경되었는지 확인
ls data
```

\- 주의할 점은 데이터 버전 변경을 git commit의 COMMIT\_HASH 값으로 변경하기 때문에 data.dvc 파일을 commit 할때 신중하게 작성해야 한다.
\- 특정 파일의 git commit 이력을 확인할 때 git log -p <filename>

<br>

#### **5.  Github**

\- GitHub에 코드와 dvc정보를 올려두면 Dockfile에서 git clone만으로 데이터를 쉽게 가져올 수 있게된다.
\- ssh파일을 통해 연동되기 때문에 반드시 생성한 ssh private key를 포함시켜서 git에 push 해주어야 한다.
\- git remote 사용법은 생략

```  python
git push
```

#### **6\. Docker**

1) Dockerfile 작성

``` python
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 설치 가능한 리스트 업데이트
RUN apt-get update && apt-get install -y git

# dvc install
RUN pip install dvc[all]==2.6.4

RUN git clone {github/repository/path} dvc-tutorial

WORKDIR dvc-tutorial

# 데이터 다운로드
RUN dvc pull
```

<br>


2) 이미지 빌드

\- 명령어를 실행하는 경로에 Dockerfile 있으면 맨 뒤에 .

``` python
docker build -t dvc-tutorial:v1.0.0 .
```

<br>

3) 컨테이너 실행

\- /bin/bash를 통해 컨테이너 bash shell 진입

\- 코드와 데이터가 잘 다운로드 되었는지 확인

``` python
docker run -it dvc-tutorial:v1.0.0 /bin/bash

ls data
```

---

#### **End**

이번 포스팅에서는 git과함께 DVC를 통해 데이터 버전관리를 해보고 Docker 이미지빌드에 활용해 보았다. 머신러닝 프로젝트를 진행하면 코드는 Github에서 관리할 수 있지만 용량이 큰 데이터는 무료로 Github에서 관리할 수 없다. 

반면에 DVC는 무료로 사용 가능하며 쉽게 적용할 수 있다는 장점이 있고 더불어 MLOps의 일부를 지원하기 때문에 머신러닝 프로젝트를 진행한다면 사용해볼 가치가 있다고 생각한다.

Reference

DVC - [https://dvc.org/](https://dvc.org/)  
DVC ssh 활용법 - [https://discuss.dvc.org/t/how-do-i-use-dvc-with-ssh-remote/279/2](https://discuss.dvc.org/t/how-do-i-use-dvc-with-ssh-remote/279/2)