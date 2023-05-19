---
title:  "AWS 기초 서비스 정리"
folder: "aws"
categories:
  - aws
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
---

## IAM
![](/images/aws-IAM.png){: .align-center height="40%" width="40%"}

> 공식문서

AWS Identity and Access Management(IAM)를 사용하면 AWS 서비스와 리소스에 대한 액세스를 안전하게 <font color=pink>관리</font>할 수 있습니다. 또한, AWS 사용자 및 그룹을 만들고 관리하며 AWS 리소스에 대한 액세스를 <font color=pink>허용 및 거부</font>할 수 있습니다. (언제 어디서 누가 무엇을 어떻게)

<br>

> 요약

- AWS acount 관리 및 리소스/사용자/서비스의 권한 제어
	- 서비스 사용을 위한 인증 정보 부여
- 사용자의 생성,관리 및 계정의 보안
	- 사용자의 패스워드 정책 관리 (일정 시간마다 패스워드 변경 등)
- 다른 계정과의 리소스 공유
	- Identity Federation (Facebook 로그인, 구글 로그인 등)
- 계정에 별명 부여 가능 -> 로그인 주소 생성 가능
- IAM은 글로벌 서비스 (Region 서비스 X)

<br>

> IAM 구성 4가지
 
사용자
- 실제 AWS를 사용하는 사람 혹은 어플리케이션을 의미

그룹
- 사용자의 집합
- 그룹에 속한 사용자는 그룹에 부여된 권한을 행사

정책(Policy)
- 사용자와 그룹, 역할이 무엇을 할 수 있는지에 관한 문서
- JSON(JavaScript Object Notation) 형식으로 정의

역할(Role)
- AWS 리소스에 부여하여 AWS 리소스가 무엇을 할 수 있는지를 정의
- 혹은 다른 사용자가 역할을 부여 받아 사용
- 다른 자격에 대해서 신뢰관계를 구축 가능
- 역할을 바꾸어 가며 서비스를 사용 가능

<br>

>  역할과 정책의 차이

<style>
table th:first-of-type {
    width: 10%;
}
table th:nth-of-type(2) {
    width: 10%;
}
table th:nth-of-type(3) {
    width: 50%;
}
table th:nth-of-type(4) {
    width: 30%;
}
</style>

| 정책                                | 역할                      |
| :-----------------------------------: | :-------------------------: |
| 하나의 IAM 사용자, 그룹 단위에 부여 | 역할을 부여받은 모든 대상 |
| 장기 또는 영구적 자격 증명          | 임시 보안 자격 증명       |
| 정책을 부여받은 동안 권한 획득      | 특정시간 동안만 권한 획득                          |


<br>

> IAM 자격 증명 보고서

- 계정의 모든 사용자와 암호, 엑세스 키, MFA(Multi-Factor Authentication) 장치등의 증명 상태를 나열하는 보고서를 생성하고 다운로드 가능
- 4시간에 한번 씩 생성 가능
- AWS 콘솔, CLI, API에서 생성 요청 및 다운로드 가능

<br>

> IAM 모범 사용 예시

- 루트 사용자는 사용하지 않기
- 불필요한 사용자는 만들지 않기(관리가 어려움)
- 가능하면 그룹과 정책을 사용하기
- 최소한의 권한만을 허용하는 습관을 들이기 (Principle of least privilege)
- MFA를 활성화 하기
- AccessKey 대신 역할을 활용하기
- IAM 자격 증명 보고서(Credential Report) 활용하기

<br>

## EC2

![](/images/ec2.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon Elastic Compute Cloud(EC2)는 안전하고 크기 조정이 가능한 컴퓨팅 파워를 클라우드에서 제공하는 웹 서비스입니다. 개발자가 더 쉽게 웹 규모의 클라우드 컴퓨팅 작업을 할 수 있도록 설계되었습니다. Amazon EC2의 간단한 웹 서비스 인터페이스를 통해 간편하게 필요한 용량을 얻고 구성할 수 있습니다. 컴퓨팅 리소스에 대한 포괄적인 제어권을 제공하며, Amazon의 검증된 컴퓨팅 인프라에서 실행할 수 있습니다.

<br>

> EC2의 사용

- 서버를 구축할 때
	- 게임서버, 웹서버, 어플리케이션 서버
- 어플리케이션을 사용하거나 호스팅할 때
	- 데이터베이스
	- 머신러닝
	- 비트코인 채굴
	- 연구용 프로그램
- 기타 다양한 목적
	- 그래픽 렌더링
	- 게임 등

<br>

> EC2의 특성

- 초 단위 on-demand 가격 모델
	- on-demand 모델에서는 가격이 초 단위로 결정
	- 서비스 요금을 미리 약정하거나 선입금이 필요 없음
- 빠른 구축 속도와 확장성
	- 몇 분이면 전 세계에 인스턴스 수백여대를 구축 가능
- 다양한 구성방법 지원
	- 머신러닝, 웹서버, 게임서버, 이미지처리 등 다양한 용도에 최적화된 서버 구성 가능
	- 다양한 과금 모델 사용 가능
- 여러 AWS 서비스와 연동
	- 오토스케일링, Elastic Load Balancer(ELB), CloudWatch

<br>

> EC2의 구성

- 인스턴스
	- 클라우드에서 사용하는 가상 서버로 CPU, 메모리, 그래픽카드 등 연산을 위한 하드웨어를 담당
- Elastic Block Storage(EBS)
	- 클라우드에서 사용하는 가상 하드디스크
- Amazon Machine Image(AMI)
	- EC2 인스턴스를 실행하기 위한 정보를 담고 있는 이미지
- 보안 그룹
	- 가상의 방화벽

<br>

### EC2의 가격정책

> On-Demand

- 실행하는 인스턴스에 따라 시간 또는 초당 컴퓨팅 파워로 측정된 가격을 지불
- 약정은 필요없음
- 장기적인 수요 예측이 어렵거나 유연하게 EC2를 사용하고 싶을 때
- 한번 써보고 싶을 때

<br>

> 예약 인스턴스(Reserved Instance - RI)

- 미리 일정 기간(1~3년) 약정해서 쓰는 방식
- 최대 75% 정도 저렴
- 수요 예측이 확실할 때
- 총 비용을 절감하기 위해 어느정도 기간의 약정이 가능한 사용자

<br>

> Spot Instance

- 경매 형식으로 시장에 남는 인스턴스를 저렴하게 구매해서 쓰는 방식
	- 수요에 따라 스팟 인스턴스의 가격은 계속 변동
	- 내가 지정한 가격보다 현재 가격이 낮다면 사용
	- 내가 지정한 가격보다 현재 가격이 높다면 반환
- 최대 90% 정도 저렴
- 단, 언제 도로 내주어야 할지 모름(예측 불가능)
	- 인스턴스가 확보되고 종료되는 것을 반복해도 문제 없는 분산 아키텍쳐 필요
- 주로 빅데이터 처리, 머신러닝 등 많은 인스턴스가 필요한 작업에 사용

<br>

> 전용 호스트(Dedicated)

- 가상화된 서버에서 EC2를 빌리는 것이 아닌 실제 물리적인 서버를 임대하는 방식
- 라이선스 이슈 (Windows Server 등)
- 보안이 엄격히 이루어저야 하는 경우
- 퍼포먼스 이슈 (CPU Steal 등)
	- CPU Steal이란 하나의 물리적 서버내에 여러개의 가상 서버가 돌고 있을때 가상 서버끼리의 CPU간섭이 일어나는 현상
	- 즉 CPU 사용량을 정확하게 제어하고 싶을 때 사용하기도 함

<br>

> 기타

- 가격순서
	- 스팟 인스턴스 < 예약 인스턴스 < 온디맨드 < 전용 호스트
- EC2의 가격모델은 EBS와는 별도
	- EBS는 사용한 만큼 지불
- 기타 데이터 통신 등의 비용은 별도로 청구
	- 참고로 AWS는 AWS 바깥으로 나가는 트래픽에 대해서만 요금을 부과

<br>

### EBS

![](/images/ebs.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon Elastic Block Store(EBS)는 AWS 클라우드의 Amazon EC2 인스턴스에 사용할 영구 블록 스토리지 볼륨을 제공합니다. 각 Amazon EBS 볼륨은 가용 영역 내에 자동으로 복제되어 구성요소 장애로부터 보호해주고, 고가용성 및 내구성을 제공합니다. Amazon EBS 볼륨은 워크로드 실행에 필요한 지연 시간이 짧고 일관된 성능을 제공합니다. Amazon EBS를 사용하면 단 몇 분 내에 사용량을 많게 또는 적게 확장할 수 있으며, 프로비저닝한 부분에 대해서만 저렴한 비용을 지불합니다.

<br>

> 요약

- 가상 하드드라이브
- EC2 인스턴스가 종료되어도 계속 유지 가능
	- EC2 인스턴스와  EBS는 네트워크로 연결되어있기 때문
	- 인스턴스 업그레이드에 용이
	- 하나의 EC2 인스턴스가 여러개의 EBS를 가질 수 있음(반대도 가능)
- 인스턴스 정지 후 재 기동 가능
-  루트 볼륨으로 사용시 EC2가 종료되면 같이 삭제됨
	- 단 설정을 통해 EBS만 따로 존속 가능
- <font color=pink>EC2와 같은 가용영역에 존재</font>
- 총 5가지 타입을 제공
	- 범용 (General Purpose or GP3) : SSD
	- 프로비저닝 된 IOPS (Provisioned IOPS or io2) : SSD
	- 쓰루풋 최적화 (Throughput Optimized HDD or st1)
	- 콜드 HDD (SC1)
	- 마그네틱 (Standard)

<br>

### Snapshot

![](/images/snapshot.png){: .align-center height="40%" width="40%"}

> 요약

- 특정시간의 EBS 상태 저장본
	- EBS의 사진을 찍어둔 개념
- 필요시 스냅샷을 통해 특정 시간의 EBS를 복구 가능
- S3에 보관
	-  증분식 저장 (변화된 부분만 저장)

<br>

### AMI

![](/images/AMI.png){: .align-center height="40%" width="40%"}

> 요약

- EC2 인스턴스를 실행하기 위해 필요한 정보를 모은 단위
	- OS, 아키텍쳐 타입(32-bit or 64-bit), 저장공간 용량 등
- AMI를 사용하여 EC2를 복제하거나 다른 리전 -> 계정으로 전달 가능
- 구성
	- 1개 이상의 EBS 스냅샷
	- 인스턴스 저장의 경우 루트 볼륨에 대한 템플릿(예: 운영 체제, 애플리케이션 서버, 애플리케이션)
	- 사용권한 (어떤 AWS 어카운트가 사용할 수 있는지)
	- 블록 디바이스 맵핑(EC2 인스턴스를 위한 볼륨 정보 = EBS가 무슨 용량으로 몇개 붙는지)
- 타입
	- EBS기반 or 인스턴스 저장 기반
		- EBS 기반: 스냅샷을 기반으로 루트 디바이스 생성
		- 인스턴스 저장: S3에 저장된 템플릿을 기반으로 생성

<br>

### EC2의 생명주기

중지
- 중지 중에는 인스턴스 요금 미 청구
- 단 EBS요금, 다른 구성요소 (Elastic IP등)은 청구
- 중지 후 재시작시 Public IP 변경
- EBS를 사용하는 인스턴스만 중지 가능 -> 인스턴스 저장 기반의 인스턴스는 중지 불가

재부팅
- 재부팅시에는 Public IP 변동 없음

최대 절전모드
- 메모리 내용을 보존해서 재시작시 중단지점에서 시작할 수 있는 정지모드

<br>

### Autoscaling(EC2 Auto Scaling)

![](/images/Auto-scaling.png){: .align-center height="40%" width="40%"}

> 공식문서

AWS Auto Scaling은 애플리케이션을 모니터링하고 용량을 자동으로 조정하여, 최대한 저렴한 비용으로 안정적이고 예측 가능한 성능을 유지합니다. AWS Auto Scaling을 사용하면 몇 분 만에 손쉽게 여러 서비스 전체에서 여러 리소스에 대해 애플리케이션 규모 조정을 할 수 있습니다.

<br>

> 스케일링?

Vertical Scale(Scale Up)
- 성능 자체를 올리는것.
	- 메모리 1GB -> 16GB
- 좋은 메모리를 사용할수록 가격은 점점 더 비싸짐

Horizontal Scale(Scale Out)
- 규모를 늘리는 것.
	- 메모리 1GB -> 1GB x 16개 = 16GB
- 성능 향상은 동일하나 Vertical Scale 과 다르게 가격 상승이 비례

<br>

> 목표

- 정확한 수의 EC2 인스턴스를 보유하도록 보장
	- 그룹의 최소 인스턴스 숫자 및 최대 인스턴스 숫자
		- 최소 숫자 이하로 내려가지 않도록 인스턴스 숫자 유지(인스턴스 추가)
		- 최대 숫자 이상 늘어나지 않도록 인스턴스 숫자 유지(인스턴스 삭제)
	- 다양한 스케일링 정책 적용 가능
		- 예: CPU의 부하에 따라 인스턴스 크기를 늘리기
- 다양한 가용 영역에 인스턴스가 골고루 분산될 수 있도록 인스턴스를 분배

<br>

> 구성

- 시작 템플릿: 무엇을 실행시킬 것인가?
	- EC2의 타입, 사이즈
	- AMI
	- 보안 그룹, Key, IAM
	- 유저 데이터
- 모니터링: 언제 실행시킬것인가? + 상태 확인
	- 예: CPU 점유율이 일정 %를 넘어섰을 때 추가로 실행 or 2개 이상이 필요한 스택에서 EC2 하나가 죽었을 때
	- CloudWatch, ELB와 연계
- 설정: 얼마나 어떻게 실행시킬 것인가?
	- 최대 / 최소 / 원하는 인스턴스 숫자
	- ELB와 연동 등

<br>

## ELB

![](/images/ELB.png){: .align-center height="40%" width="40%"}

> 공식문서

Elastic Load Balancing은 애플리케이션에 들어오는 트래픽을 Amazon EC2 인스턴스, 컨테이너, IP 주소, Lambda 함수와 같은 여러 대상에 자동으로 분산시킵니다. Elastic Load Balancing은 단일 가용 영역 또는 여러 가용 영역에서 다양한 애플리케이션 부하를 처리할 수 있습니다. Elastic Load Balancing이 제공하는 세 가지 로드 밸런서는 모두 애플리케이션의 내결함성에 필요한 고가용성, 자동 확장/축소, 강력한 보안을 갖추고 있습니다.

<br>

> 요약

- 다수의 서비스에 트래픽을 분산 시켜주는 서비스
- Health Check: 직접 트래픽을 발생시켜 Instance가 살아있는지 체크
- Autoscaling 과 연동 가능
- 여러 가용영역에 분산 가능
- 지속적으로 IP 주소가 바뀌며 IP 고정 불가능 : <font color=pink>항상 도메인 기반으로 사용</font>
- 총 4가지 종류
	- Application Load Balancer(ALB) -> 가장 많이 쓰이는 로드밸런서
	- Network Load Balancer -> TCP 기반 빠른 트래픽 분산, Elastic IP 할당 가능
	- Classic Load Balancer -> 오래된 로드밸런서
	- Gateway Load Balancer -> 먼저 트래픽을 체크(인증, 로깅, 캐싱, 방화벽)하는 로드밸런서

<br>

> 대상 그룹(Target Group)

- ALB가 라우팅 할 대상의 집합
- 구성
	- 3+1 가지 종류
		- Instance
		- IP (public IP는 불가능)
		- Lambda
		- ALB (다른 로드밸런서와도 연동 가능)
	- 프로토콜 (HTTP, HTTPS, gRPC 등)
	- 기타 설정
		- 트래픽 분산 알고리즘, 고정 세션 등 

<br>

## EFS

![](/images/EFS.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon EFS(Amazon Elastic File System)는 AWS 클라우드 서비스와 온프레미스 리소스에서 사용할 수 있는, 간단하고 확장 가능하며 탄력적인 완전관리형 NFS 파일 시스템을 제공합니다. 이 제품은 애플리케이션을 중단하지 않고 온디맨드 방식으로 페타바이트 규모까지 확장하도록 구축되어, 파일을 추가하고 제거할 때 자동으로 확장하고 축소하며 확장 규모에 맞게 용량을 프로비저닝 및 관리할 필요가 없습니다.

<br>

> 요약

- NFS 기반 공유 스토리지 서비스(NFSv4)
	- 따로 용량을 지정할 필요 없이 사용한 만큼 용량이 증가 <-> EBS는 미리 크기를 지정해야함
	- 페타바이트 단위까지 확장 가능
	- 몇 천개의 동시 접속 유지 가능
	- 데이터는 여러 AZ(가용영역)에 나누어 분산 저장
	- 쓰기 후 읽기(Read After Write) 일관성
	- Private Service: AWS 외부에서 접속 불가능
		- AWS외부에서 접속하기 위해서는 VPN 혹은 Direct Connect 등으로 별도로 VPC와 연결 필요
	- 각 가용역역에 Mount Target을 두고 각각의 가용영역에서 해당 Mount Target으로 접근
	- Linux Only

<br>

> EFS 퍼포먼스 모드

- General Purpose: 가장 보편적인 모드. 거의 대부분의 경우 사용 권장
- Max IO: 매우 높은 IOPS가 필요한 경우
	- 빅데이터, 미디어 처리 등

<br>

> EFS Throughput 모드

- Bursting Throughput: 낮은 Throughput일 때 크레딧을 모아서 높은 Throughput일 때 사용
	- EC2 T타입과 비슷한 개념
- Provisioned Throughput: 미리 지정한 만큼의 Throughput을 미리 확보해두고 사용

<br>

> EFS 스토리지 클래스

- EFS Standard : 3개 이상의 가용영역에 보관
- EFS Standard - IA : 3개 이상의 가용영역에 보관, 조금 저렴한 비용 대신 데이터를 가져올 때 비용 발생
- EFS One Zone : 하나의 가용영역에 보관 -> 저장된 가용영역의 상황에 영향을 받을 수 있음
- EFS One Zone - IA : 저장된 가용영역의 상황에 영향을 받을 수 있음, 데이터를 가져올 때 비용 발생(가장 저렴)

<br>

## 네트워크 기초


> 사설 IP (Private IP)

- 한정된 IP주소를 최대한 활용하기 위해 IP주소를 분할하고자 만든 개념
	- IPv4 기준으로 최대 IP갯수는 약 43억개
- 사설망
	- 사설망 내부에는 외부 인터넷 망으로 통신이 불가능한 사설 IP로 구성
	- 외부로 통신할 때는 통신 가능한 공인 IP로 나누어 사용
	- 보통 하나의 망에는 사설 IP를 부여받은 기기들과 NAT 기능을 갖춘 Gateway로 구성
- 참고: IPv6 최대 IP개수: 2^128
	- IPv4에 비교하면 IP주소가 매우 많기 때문에 사설IP에 대한 고려가 필요 없음

<br>

> NAT(Network Address Translation)

- 사설 IP가 공인 IP로 통신할 수 있도록 주소를 변환해 주는 방법
- 3가지 종류
	- Dynamic NAT: 1개의 사설 IP를 가용 가능한 공인 IP로 연결
		- 공인 IP 그룹(NAT Pool)에서 현재 사용 가능한 IP를 가져와서 연결
	- Static NAT: 하나의 사설 IP를 고정된 하나의 공인 IP로 연결
		- AWS Internet Gateway가 사용하는 방식
	- PAT(Port Address Translation): 많은 사설 IP를 하나의 공인 IP로 연결
		- NAT Gateway / NAT Instance가 사용하는 방식
		- 일반적으로 회사에서 사용하는 방식

<br>

> CIDR(Classless Inter Domain Routing)

- IP는 주소의 영역을 여러 네트워크 영역으로 나누기 위해 IP를 묶는 방식
- 여러 개의 사설망을 구축하기 위해 망을 나누는 방법

<br>

> CIDR Block/CIDR Notation

- CIDR Notation : CIDR Block을 표시하는 방법
	- 네트워크 주소와 호스트 주소로 구성
	- 각 호스트 주소 숫자 만큼의 IP를 가진 네트워크 망 형성 가능
	- A.B.C.D/E 형식
		- 예: 10.0.1.0/24 , 172.16.0.0/12
		- A,B,C,D: 네트워크 주소 + 호스트 주소 표시 / E: 0~32: 네트워크 주소가 몇 bit인지 표시

- CIDR Block: IP 주소의 집합
	- 호스트 주소 비트만큼 IP 주소를 보유 가능
	- 예: 192.168.2.0/24
		- 네트워크 비트 24
		- 호스트 주소 = 32 - 24 = 8
		- 즉 2^8 = 256개의 IP 주소 보유
		- 192.168.2.0 ~ 192.168.2.255 까지 총 256개 주소를 의미

<br>

> 서브넷

- 네트워크 안의 네트워크
- 큰 네트워크를 잘게 쪼갠 단위
- 일전 IP주소의 범위를 보유
	- 큰 네트워크에 부여된 IP범위를 조금씩 잘라 작은 단위로 나눈 후 각 서브넷에 할당
- 예시
	- 네트워크 - 192.168.0.0/16
	- 서브넷 1 - 192.168.2.0/24, 서브넷 2 - 192.168.3.0/24, 서브넷 3 - 192.168.4.0/24
	- 서브넷 1은 192.168.2.1부터 192.168.2.255 까지의 IP주소를 가지고 있음
- 서브넷을 잘게 자를 때 필요한 것이 CIDR

<br>

> AWS의 네트워크 구조

- 외부에서 각종 AWS서비스에 접근이 가능하지만 원칙적으로 VPC는 외부에서 접근이 불가능함.
- VPC내부의 EC2에서 S3등과 같은 서비스로 바로 접근이 불가능하다는 의미.
- 하지만 이것이 가능한 이유는 Internet gateway를 통해 외부로 나갔다가 s3로 다시 접근하는 방식으로 설계되었기 때문

<br>



## VPC

![](/images/VPC.png){: .align-center height="40%" width="40%"}

> 공식문서

Virtual Private Cloud(VPC)는 사용자의 AWS 계정 전용 가상 네트워크입니다. VPC는 AWS 클라우드에서 다른 가상 네트워크와 논리적으로 분리되어 있습니다. Amazon EC2 인스턴스와 같은 AWS 리소스를 VPC에서 실행할 수 있습니다. IP 주소 범위와 VPC 범위를 설정하고 서브넷을 추가하고 보안 그룹을 연결한 다음 라우팅 테이블을 구성합니다.

<br>

> 요약

- VPC = 가상으로 존재하는 데이터 센터
- 외부에 격리된 네트워크 컨테이너(유닛) 구성 가능
	- 원하는 대로 사설망을 구축 가능
	- 부여된 IP 대역을 분할하여 사용 가능
- 리전 단위

<br>

> VPC의 사용 사례

- EC2, RDS, Lambda등의 AWS의 컴퓨팅 서비스 실행
- 다양한 서브넷 구성
- 보안 설정(IP Block, 인터넷에 노출되지 않는 EC2등 구성)

<br>

> VPC의 구성요소

- 서브넷
- 인터넷 게이트웨이
- NACL/보안그룹
- 라우트 테이블
- NAT Instance/ NAT Gateway
- Bastion Host
- VPC Endpoint

<br>

### 서브넷

![](/images/Subnet.png){: .align-center height="40%" width="40%"}

- VPC의 하위 단위로 VPC에 할당된 IP를 더 작은 단위로 분할한 개념
- <font color= pink>서브넷은 가용영역(AZ)안에 위치</font>
- CIDR block range로 IP 주소 지정

<br>

> AWS 서브넷의 IP 갯수

- AWS의 사용 가능 IP숫자는 5개를 제외하고 계산
- 예: 10.0.0.0/24라면
	- 10.0.0.0: 네트워크 address
	- 10.0.0.1: VPC Router
	- 10.0.0.2: DNS Server
	- 10.0.0.3: 미래에 사용을 위해 남겨 둠
	- 10.0.0.255: 네트워크 브로드캐스트 address(단 AWS VPC는 브로드캐스트를 지원하지 않음)
	- 즉 총 사용 가능한 IP 갯수는 2^8 - 5 = 251

<br>

> 서브넷의 종류

- 퍼블릭 서브넷: 외부에서 인터넷을 통해 연결할 수 있는 서브넷
	- 인터넷 게이트웨이(igw)를 통해 외부의 인터넷과 연결되어 있음
	- 안에 위치한 인스턴스에 퍼블릭 IP부여 가능
	- 웹서버, 어플리케이션 서버 등 유저에게 노출되어야 하는 인프라
- 프라이빗 서브넷: 외부에서 인터넷을 통해 연결할 수 없는 서브넷
	- 외부 인터넷으로 경로가 없음
	- 퍼블릭 IP 부여 불가능
	- 데이터베이스, 로직 서버 등 외부에 노출 될 필요가 없는 인프라

<br>

### 라우트 테이블(Route Table)

![](/images/Route table.png){: .align-center height="40%" width="40%"}

- 트래픽이 어디로 가야 할지 알려주는 이정표
- VPC 생성시 기본으로 하나 제공
- 서브넷 단위로 붙여줄 수 있음

<br>

> 예시

| Destination   | Target                |
| :-------------: | :---------------------: |
| 10.0.0.0/16   | local                 |
| 172.31.0.0/16 | pcx-11223344556677889 |
| 0.0.0.0/0     | igw-3830de61          |

- 만약 10.0.1.231 IP가 들어오게된다면 10.0.0.0/16 과 0.0.0.0/0 에 매칭가능 
- 이때 CIDR의 E가 가장 클때 가장 구체적이다 라고 표현하고 이곳을 먼저 매칭
- 즉 10.0.1.231이라는 주소가 들어오면 라우팅 테이블을 통해 이중 가장 구체적인 local에 매칭됨

<br>

### 인터넷 게이트웨이

![](/images/Internet gateway.png){: .align-center height="40%" width="40%"}

- VPC가 외부의 인터넷과 통신할 수 있도록 경로를 만들어주는 리소스
- 1개의 VPC당 1개의 인터넷 게이트웨이 사용
- 기본적으로 확장성과 고가용성이 확보되어있음
- IPv4, IPv6 지원
	- IPv4의 경우 NAT 역할
- Route Table에서 경로 설정 후에 접근 가능
- 무료

<br>

### 보안 그룹(Security Group)

- Network Access Control List(NACL)와 함께 방화벽의 역할을 하는 서비스
- Port 허용
	- 기본적으로 모든 포트는 비활성화(접근 불가능한 상태)
	- 선택적으로 트래픽이 지나갈 수 있는 Port와 Source를 설정 가능
	- Deny(특정IP 거부 등)는 불가능 -> NACL 로 가능함
- 인스턴스 단위
	- 하나의 인스턴스에 하나 이상의 SG설정 가능
	- NACL의 경우 서브넷 단위
	- 설정된 인스턴스는 설정한 모든 SG의 룰을 적용 받음
		- 기본 5개, 최대 16개

<br>

> 보안 그룹의 Stateful

- 보안 그룹은 Stateful
- Inbound로 들어온 트래픽이 별 다른 Outbound 설정 없이 나갈 수 있음
- 즉 요청을 보낸 클라이언트의 정보(임시 포트)를 알고있기때문에 Outbound 설정이 필요없다는 의미
- NACL은 Stateless
	- 들어올때는 들어올때고 나갈때는 나갈때다. 나가려면 한번 더 확인이 필요함
	- 이때 Outbound가 설정이 되어있지 않다면 time out, connection refuse가 일어남
	- outbound는 well known port(80, 443, 22 등)나 Ephemeral port(임시포트)의 범위를 지정

<br>

### NACL

![](/images/Nacl.png){: .align-center height="40%" width="40%"}

- 보안그룹처럼 방화벽 역할을 담당
- 서브넷 단위
- 포트 및 아이피를 직접 Deny 가능
	- 외부 공격을 받는 상황 등 특정 IP를 블록하고 싶을 때 사용
- 순서대로 규칙 평가(낮은 숫자부터)

<br>

> NACL 규칙

- 규칙번호: 규칙에 부여되는 고유 숫자이며 규칙이 평가되는 순서. (낮은 번호부터)
	- AWS 추천은 100단위 증가 -> 중간에 새로운 규칙이 들어가기 편하게 하기 위함
	- 하나의 규칙이라도 만족하면 패스됨 -> 특정 IP를 차단하려면 낮은 번호로 규칙 생성해야함
- 유형: 미리 지정된 프로토콜. 선택 시 AWS에서 잘 알려진 포트(Well Known Port)이 규칙에 지정됨
- 프로토콜: 통신 프로토콜. (예: TCP, UDP, SMP ...)
- 포트 범위: 허용 혹은 거부할 포트 범위
- 소스: IP주소의 CIDR 블록
- 허용/거부: 허용 혹은 거부 여부


<br>

### NAT Gateway

![](/images/NAT gateway.png){: .align-center height="40%" width="40%"}

> 공식문서

 VPC의 프라이빗 서브넷에 있는 인스턴스에서 인터넷에 쉽게 연결할 수 있도록 지원하는 고가용성 관리형 서비스입니다.

<br>

> NAT Gateway/NAT Instance

- Private 인스턴스가 외부의 인터넷과 통신하기 위한 통로
- NAT Instance는 다른 대상에게 트래픽을 중계해주는 단일 EC2 인스턴스 
- NAT Gateway는 AWS에서 제공하는 서비스
- NAT Gateway/Instance는 모두 서브넷 단위
	- Public Subnet에 있어야 함

<br>

> Bastion Host

- Private 인스턴스에 접근하기 위한 EC2 인스턴스
- Public 서브넷에 있어야 함(key-pair 등록)
- 신경써야 하는 부분이 많아서 Session Manager 서비스로 Private EC2로의 접속을 대체 가능

<br>

## ENI

![](/images/ENI.png){: .align-center height="40%" width="40%"}

> 공식문서

Elastic Network Interface는 VPC에서 가상 네트워크 카드를 나타내는 논리적 네트워킹 구성 요소입니다.

<br>

> 요약

- EC2의 가상의 랜카드
	- IP주소와 MAC 주소를 보유
	- ENI 하나 당 Private IP + 하나의 Public IP(Optional)
	- 필요에 따라서 한 개 이상의 Private IP 부여 가능
-  EC2는 반드시 하나 이상의 ENI가 연결되어 있음
	- 제일 처음 EC2를 생성할 때 Primary ENI가 생성되어 연결됨
	- 즉 하나의 EC2는 하나 이상의 ENI 보유 가능
	- 추가적인 ENI의 경우 EC2와 같은 가용영역(AZ)이면 다른 서브넷에도 설정 가능
- <font color= pink>실질적으로 EC2는 가용영역안에 만들어지고 어떤 서브넷을 사용할지, 보안그룹 설정 등 외부와 관련된 연결은 ENI 단위에서 결정. 즉 EC2 인스턴스는 서브넷 안에 속해있지 않고 EC2에 설정된 ENI가 속해있음 </font>

<br>

> 다중 ENI 아키텍쳐

- 하나의 EC2 인스턴스에 여러 ENI를 연동 가능
- 사용 사례
	- ENI 교체를 통한 배포/업데이트
	- 관리를 위하여 하나의 EC2 인스턴스에 다양한 접근 경로 설정
		- 다양한 서브넷에서 EC2 인스턴스에 접근하고싶을 때
	- MAC address에 종속된 라이선스 프로그램을 다양한 EC2에서 사용
- 동시에 연동 가능한 ENI 숫자는 EC2의 타입과 크기에 따라 다름(t2-micro는 2개)

<br>

> ENI와 보안 그룹

- 보안그룹 적용은 ENI 단위
	- 즉 하나의 EC2 인스턴스에 다양한 보안 그룹으로 구성된 경로를 적용가능
	- 예: Subnet A에서는 80번만 허용, Subnet B에서는 22번만 허용
- NACL은 Subnet 단위이기 때문에 관계없음

<br>

> EC2와 Public IP

- EC2 Public IP는 ENI가 아닌 가상의 Public IP <-> Private IP 테이블로 관리됨
	- ENI가 Public IP를 가지는게 아님
	- 이 레코드는 Elastic IP로 고정하지 않는 이상 영구적인 레코드가 아님
	- EC2의 중지 -> 재부팅 시 Private IP는 바뀌지 않으나, Public IP는 바뀌는 이유
- 인터넷에서 Public IP로 통신이 전달되면 IGW가 이 테이블을 통해 변환 후 전달
	- 즉 EC2의 OS는 절대로 Public IP를 알 수 없음
	- Private IP의 경우 OS에서 확인 가능
- EC2를 생성할 때 만들어지는 Primary ENI가 아닌 경우에 Public IP를 부여하려면 Elastic IP의 활용 필요

<br>

> Source/Destination Check

- ENI는 기본적으로 자신이 발생시켰거나, 자신이 대상이 아닌 트래픽은 무시
- 단 설정에 따라서 해제 가능
	- NAT Instance등 자신을 위한 트래픽이 아닌 다른 대상에게 중계해주는 경우 해제 필요
- ENI 단위 


<br>

## Amazon S3

![](/images/S3.png){: .align-center height="40%" width="40%"}


> 공식문서

Amazon Simple Storage Service(Amazon S3)는 업계 최고의 확장성과 데이터 가용성 및 보안과 성능을 제공하는 객체 스토리지 서비스

99.9999%의 내구성을 제공하도록 설계되었으며, 전 세계 기업의 수많은 애플리케이션을 위한 데이터를 저장

<br>

> 객체 스토리지 서비스

- 파일 보관만 가능
- 어플리케이션 설치 불가능
- <font color= pink>글로벌 서비스 단, 데이터는 리전에 저장</font> 
- 무제한 용량(하나의 객체는 0byte~5TB 용량)

<br>

> bucket

- S3의 저장공간을 구분하는 단위
- 디렉토리와 같은 개념
- bucket 이름은 전 세계에서 고유 값: 리전에 관계 없이 중복된 이름이 존재할 수 없음

<br>

> S3 객체의 구성

- Owner: 소유자
- Key: 파일의 이름
- Value: 파일의 데이터
- Version Id: 파일의 버전 아이디
- Metadata: 파일의 정보를 담은 데이터
	- 파일이 0byte 더라도 Metadata를 저장하여 사용할 수도 있음
- ACL: 파일의 권한을 담은 데이터 
- Torrents: 토렌트 공유를 위한 데이터

<br>

> S3의 내구성

- 최소 3개의 가용영역(AZ)에 데이터를 분산 저장(Standard의 경우)
- 즉 파일을 잃어버릴 확률이 매우 낮음
- 99.9% SLA 가용성(스토리지 클래스에 따라 다름)

<br>

> 보안설정

-  S3 모든 버킷은 새로 생성시 기본적으로 Private
	- 따로 설정을 통해 불특정 다수에게 공개 가능 (i.e 웹 호스팅)
- 보안 설정은 객체 단위와 버킷 단위로 구성
	- Bucket Policy: 버킷 단위
	- ACL(Access Control List): 객체 단위 (안쓰이는 추세)
- MFA를 활용해 객체 삭제 방지 가능
- Versioning을 통해 파일 관리 가능
- 액세스 로그 생성 및 전송 가능
	- 다른 버킷 혹은 다른 계정으로 전송 가능

<br>

### S3 스토리지 클래스

- S3는 다양한 스토리지 클래스를 제공
	- 클래스별로 저장의 목적, 예산에 따라 다른 저장 방법을 적용
	- 총 8가지 클래스

<br>

> S3 스탠다드

- 99.99$ 가용성
- 99.999999999% 내구성
- 최소 3개 이상의 가용영역에 분산 보관
- 최소 보관 기간 없음, 최소 보관 용량 없음
- 파일 요청 비용 없음 (전송 요금은 발생)
- $0.025/gb(Seoul Region 기준

<br>

> S3 스탠다드 IA(Infrequently Accessed)

- 자주 사용되지않는 데이터를 저렴한 가격에 보관
- 최소 3개 이상의 가용영역에 분산 보관
- 최소 저장 용량: 128kb
	- 5kb를 저장해도 128kb로 취급하여 요금 발생
- 최소 저장 기간: 30일
	- 즉 1일만 저장해도 30일의 요금 발생
- 데이터 요청 비용 발생: 데이터를 불러올 때마다 비용 지불(per GB)
- 사용 사례: 자주 사용하지 않는 파일 중 중요한 파일
- $0.0138/gb(Seoul Region 기준)

<br>

> S3 One Zone-IA

- 자주 사용되지 않고, 중요하지 않은 데이터를 저렴한 가격에 보관
- <font color= pink>단 한 개의 가용 영역에만 보관</font>
- 최소 저장 용량: 128kb
- 최소 저장 기간: 30일
- 데이터 요청 비용 발생: 데이터를 불러올 때마다 비용 지불(per GB)
- 사용 사례: 자주 사용하지 않으며 쉽게 복구할 수 있는 파일
- 0.011/gb(Seoul Region 기준)

<br>

> S3 Glacier Instant Retrieval

- 즉각적으로 가져올수 있는 빙산?
- 아카이브용 저장소
- 최소 저장 용량 : 128kb
- 최소 저장 기간 : 90일
- 바로 액세스 가능
- 사용 사례: 의료 이미지 혹은 뉴스 아카이브 등
- $0.005/gb(Seoul Region 기준) = Standard의 약1/5

<br>

> S3 Glacier Flexible Retrieval

- 아카이브용 저장소
- 최소 저장 용량 : 40kb
- 최소 저장 기간: 90일
- 분~ 시간 단위 이후 액세스 가능
- 사용 사례: 장애 복구용 데이터, 백업 데이터 등
- $0.0045/gb(Seoul Region 기준)

<br>

> S3 Glacier Deep Archive

- 아카이브용 저장소
- 최소 저장 용량: 40kb
- 최소 저장 기간: 90일
- 데이터를 가져오는데 12~48시간 소요
- 사용 사례: 오래된 로그 저장, 사용할 일이 거의 없지만 법적으로 보관해야 하는 서류 등

<br>

> S3 Intelligent-Tiering

- 머신러닝을 사용해 자동으로 S3 스토리지 클래스 변경
	- 파일 사용패턴을 분석해서 적절한 클래스로 변경해줌
- 퍼포먼스 손해/오버헤드 없이 요금 최적화

<br>

> S3 on Outposts(서비스에 가까움)

- 온프레미스 환경에 S3 제공
- IAM, S3 SDK등 사용 가능

<br>

## Serverless

서버리스(serverless)란 개발자가 서버를 관리할 필요 없이 애플리케이션을 빌드하고 실행할 수 있도록 하는 클라우드 네이티브 개발 모델입니다.

서버리스 모델에서도 서버가 존재하긴 하지만, 애플리케이션 개발에서와 달리 추상화되어 있습니다. 클라우드 제공업체가 서버 인프라에 대한 프로비저닝, 유지 관리, 스케일링 등의 일상적인 작업을 처리하여, 개발자는 배포를 위해 코드를 패키징하기만 하면 됩니다.

<br>

> AWS의 Serverless

- 서버의 관리와 프로비전 없이 코드를 실행할 수 있음
- 사용한 만큼만 비용을 지불 (OnDemand)
- 고가용성과 장애 내구성이 확보되어 있음
- 빠르게 배포하고 업데이트 기능
- Serverless 환경을 잘 활용할 수 있는 아키텍쳐 필요
	- 병렬 처리
	- 이벤트 기반 아키텍쳐 등

<br>

### SQS

![](/images/SQS.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon Simple Queue Service(SQS)는 마이크로 서비스, 분산 시스템 및 서버리스 애플리케이션을 쉽게 분리하고 확장할 수 있도록 지원하는 완전관리형 메시지 대기열 서비스입니다.

<br>

> 요약

- AWS에서 제공하는 큐 서비스
	- 다른 서비스에서 사용할 수 있도록 메시지를 잠시 저장하는 용도
	- 최대 사이즈 256kb, 최대 14일까지 저장 가능
- 주로 AWS 서비스들의 느슨한 연결(디커플링)을 수립하기 위해 사용
	- 앞 뒤로 다른 서비스가 붙어도 각 서비스는 SQS의 메시지만 처리하면 되기 때문에 문제 없음
	- 즉 확장성이 좋음
- <font color= pink>하나의 메시지를 한번만 처리</font>
- AWS에서 제일 오래된 서비스

<br>

### SNS

![](/images/SNS.png){: .align-center height="40%" width="40%"}


> 공식문서

Amazon Simple Notification Service(SNS)는 애플리케이션 간(A2A) 및 애플리케이션과 사용자 간(A2P) 통신 모두를 위한 완전관리형 메시징 서비스입니다.

<br>

> 요약

- Pub/Sub 기반의 메세징 서비스
	- 하나의 토픽을 여러 주체가 구독
	- 토픽에 전달된 내용을 구독한 모든 주체가 전달받아 처리
- 다양한 프로토콜로 메시지 전달 가능
	- 이메일
	- HTTP(S)
	- SQS
	- SMS
	- Lambda
- <font color= pink>하나의 메시지를 여러 서비스에서 처리</font> 

<br>

### Amazon EventBridge

![](/images/Eventbridge.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon EventBridge는 자체 애플리케이션, 통합 Software-as-a-Service(SaaS) 애플리케이션 및 AWS 서비스에서 생성된 이벤트를 사용하여 이벤트 기반 애플리케이션을 대규모로 손쉽게 구축할 수 있는 서버리스 이벤트 버스입니다.

<br>

> 이벤트의 특성

- 명령이 아닌 관찰한 내용
	- 명령: 생성 주체가 대상의 행동에 대한 관심을 가지고 회신을 기다림
	- 이벤트: 생성 주체는 대상의 행동에 관심에 없음 (그냥 일이 벌어진 것)
- 구성요소
	- 사건의 내용
	- 사건의 발생 시간 및 주체
	- 불변성: 과거의 생성된 이벤트는 변경될 수 없음을 보장

<br>

> AWS API Call via CloudTrail

- CloudTrail을 통해 AWS상의 모든 api call을 모니터링하고 있음
- 미리 지정된 Event 이외의 상황을 처리할 때 사용
- CloudTrail의 로그를 통해 이벤트를 발생시키는 방법
	- 즉 AWS에서 발생하는 모든 액션을 이벤트로 처리가능
- CloudTrail이 활성화 되어있어야 사용 가능
- CloudTrail과 비슷하게 CloudWatch의 로그/매트릭을 활용하여 이벤트를 트리거할 수 있음

<br>

> Amazon EventBridge 규칙

- 발생한 이벤트를 대상 서비스가 처리할 수 있도록 전달
- 다양한 대상에 동시에 전달 가능
	- API Gateway, CloudWatch Log 그룹, CodePipeline, StepFunctions, SQS, SNS등
- 두 가지 모드
	- 이벤트 패턴: AWS의 이벤트 버스에서 특정 이벤트를 패턴 매칭하여 대상에 전달
	- 스케쥴: Cron 이벤트를 활용하여 특정 시간, 혹은 주기로 대상에게 전달

<br>

> 이벤트 패턴 매칭

- AWS의 이벤트 내용 중 필요한 내용을 선별하여 패턴으로 정의
- 이후 패턴에 매칭되는 이벤트를 대상으로 보냄
- JSON 형식으로 구성
	- 매칭하고 싶은 이벤트의 내용은 Array 안에 넣어 매칭
- 일반적으로 source 필드와 detail-type 필드를 매칭하여 이벤트 종류를 분리한 후 detail 필드 안에 있는 값으로 세부 필터링

<br>

> InputTransformer

- 대상에 전달할 이벤트 내용을 편집할 수 있는 기능
	- Raw 데이터(JSON) 대신 의미있는 문장으로 전달 가능
	- 이메일 등 사람이 보는 메시지에 주로 사용
	- ex) [InstanceId]가 [State] 상태로 변경되었습니다.

<br>

### AWS Lambda

![](/images/lambda.png){: .align-center height="40%" width="40%"}

> 공식문서

AWS Lambda는 서버를 프로비저닝 또는 관리하지 않고도 실제로 모든 유형의 애플리케이션 또는 백엔드 서비스에 대한 코드를 실행할 수 있는 이벤트 중심의 Serverless 컴퓨팅 서비스입니다. 200개가 넘는 AWS 서비스와 서비스형 소프트웨어(SaaS) 애플리케이션에서 Lambda를 트리거 할 수 있으며 사용한 만큼 지불하면 됩니다.

<br>

> 요약

- AWS의 Serverless 컴퓨팅 서비스
	- 코드와 코드를 실행하기 위한 파일들을 업로드하면 서버 프로비전 없이 코드 실행
- 다양한 AWS 서비스에서 Lambda 활용가능
- 다양한 언어 지원
	- Java, C#, Go, Node.Js, Python, Ruby 등
- Lambda는 크게 두 가지 방법으로 호출
	- Event 기반
	- AWS의 다른 서비스 혹은 애플리케이션에서 직접 API Gateway를 통해서 호출
- 저렴한 가격
	- 처음 100만건 호출 무료, 이후 100만건 당 $0.2 x (메모리/컴퓨팅 사용에 따라 차이)

<br>

> AWS Lambda 구성

Deployment Package
- 함수의 코드와 코드를 실행하기 위한 런타임으로 구성
- 용량 제한
	- zip 파일: 50mb
	- unzip 파일: 250mb
	- 콘솔 에디터를 사용하려면: 3mb
- S3에 업로드 가능
- 컨테이너 이미지, Lambda Layer 등으로 제한 우회 가능

<br>

> 일반 구성(General Configuration)

- IAM 역할
- 메모리
	- 128~10,240 MB(약 10GB)
- 제한 시간
	- 최대 15분
- /tmp 디렉토리 스토리지
	- 512~10,240mb(약 10gb)

<br>

> 트리거(Trigger)

- AWS를 호출하는 서비스
	- 예:
		- API Gateway
		- SQS
		- S3
- 각 서비스에서 호출 시 지정된 양식의 이벤트 내용을 전달

<br>

> 기타

- 권한(Permissions)
	- IAM 역할로 부여된 AWS의 서비스를 사용할 수 있는 권한
- 태그
	- Lambda에 부여된 태그
- 환경변수(Environment Variables)
	- Lambda 코드상에서 사용하는 환경 변수

<br>

### Amazon API Gateway

![](/images/API Gateway.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon API Gateway는 어떤 규모에서든 개발자가 API를 손쉽게 생성, 게시, 유지관리, 모니터링 및 보안을 유지할 수 있도록 하는 완전관리형 서비스입니다. API는 애플리케이션이 백엔드 서비스의 데이터, 비즈니스 로직 또는 기능에 액세스할 수 있는 "정문" 역할을 합니다. API Gateway를 사용하면 실시간 양방향 통신 애플리케이션이 가능하도록 하는 RESTful API및 WebSocket API를 작성할 수 있습니다. API Gateway는 컨테이너식 Serverless 워크로드 및 웹 애플리케이션을 지원합니다.

<br>

> 요약

- AWS의 서비스 및 외부 서비스를 위한 API를 생성/관리해주는 서비스
- HTTP/Websocket 프로토콜 지원
- Serverless 서비스
- 다양한 AWS 서비스와 연동
	- 예: HTTP API 형식으로 Lambda 혹은 DynamodDB 연동
	- 어플리케이션 백엔드를 HTTP API로 연결
- API Key를 사용해 보안 관리와 사용량 추적 가능
- 배포 관리 가능(Canary 배포 등)

<br>

## Amazon CloudWatch

![](/images/CloudWatch.png){: .align-center height="40%" width="40%"}

> 공식문서

Amazon CloudWatch는 모니터링 및 관찰 기능 서비스입니다. CloudWatch는 애플리케이션을 모니터링하고, 시스템 전반의 성능 변경 사항에 대응하며, 리소스 사용률을 최적화하고, 운영 상태에 대한 통합된 보기를 확보하는데 필요한 데이터와 실행 가능한 통찰력을 제공합니다.

<br>

> 요약

- 어플리케이션의 모니터링 서비스
- Public 서비스
	- 인터넷을 통해 접근 가능
	- Private VPC에서는 Interface Endpoint로 접근 가능
- 로그, metric, 이벤트 등의 운영데이터를 수집하여 시각화 및 처리
	- 경보 생성을 통해 자동화된 대응 가능

<br>

> 지표(metric) 수집

 - 시간 순서로 정리된 데이터의 집합
	 - 다수의 <font color= pink>데이터 포인트</font> 로 구성
 - AWS 서비스/어플리케이션의 퍼포먼스를 모니터링 하기 위해 metric 생성
	 - EC2, Autoscaling Groups, ELB, Route53, CloudFront, EBS, Storage Gateway 등 다양한 서비스에서 기본 지원
		- 예: EC2 CPU, 네트워크, Disk IO 
	- CloudWatch Agent / API 를 활용해 커스텀 metric 생성가능
		- 유저가 원하는 데이터포인트를 생성해서 CloudWatch로 전달하여 생성 
		- 예: EC2 내부에서 알 수 있는 EC2의 메모리 사용량 등
- 리전 단위
- 15개월 이후 사라짐: 지속적으로 새로운 데이터가 들어올 경우 15개월 이전 데이터는 사라지는 형식 


<br>

> 경보(Alarm)

- 수집된 metric 값의 변동에 따라 발생하는 알림 생성
	- 일정 수치(threshold)로 도달하거나 이상/이하 일때 이벤트 발생
- 3가지 상태
	- OK: 정상상태
	- Alarm: 알람상태
	- INSUFFICIENT_DATA: 알람 상태를 확인하기 위한 정보가 부족함
- 다양한 방법으로 대응 가능
	- SNS로 Lambda 실행, 이메일 전달 등
	- 예: 웹 서버의 500에러가 일정 수치 이상일 때 슬랙 알림
- metric의 resolution에 따라 경보의 평가 주기 변동
	- High Resolution이라면 60초 미만 주기로 평가 가능
	- 이외에는 모두 60초 배수 단위로 평가


<br>

> 로그 수집 및 관리

- Lambda, EC2, Route53, ECS등 여러 AWS 서비스의 로그를 수집
- 수집된 로그를 Kinesis, S3등 다른 서비스/계정으로 전달 가능
- 혹은 자체적으로 확인하거나 쿼리 가능

<br>

> 대시보드

- 수집한 로그/metric를 기반으로 대시보드 구성
- 외부 리소스를 활용해서 커스텀 대시보드 구성 가능
	- 예: S3 객체 표시, HTML 커스텀 그래프 표시


<br>

> 기타 서비스

- 애플리케이션을 모니터링하는 Synthetics Canary
- 애플리케이션, 컨테이너, Lambda등의 문제를 찾아주는 인사이트 서비스 등

<br>

### 지표(Metric)

> metric의 구성

- 네임스페이스
	- metric의 출신 혹은 성격에 따라 논리적으로 묶은 단위
	- AWS에서 수집하는 기본적인 metric은 AWS/{서비스명} 형식
		- 예: AWS/EC2, AWS/RDS
	- 직접 metric을 생성할때에는 직접 네임스페이스를 명시해야 함. (디폴트 없음)
- metric 이름
	- metric의 고유 이름
		- 무엇에 관한 metric인지?

<br>

> 데이터 포인트(Data Points)

- metric을 구성하는 시간-값 데이터 단위
	- 시간은 초 단위까지
	- 예: 2023-05-18T22:55:59Z
- UTC 기준이 매우 권장됨
	- 내부적인 통계 혹은 알람 등에서 UTC 기준으로 활용

<br>

> 데이터 포인트의 Resolution

- 데이터가 얼마나 자주 소집되는지를 나타내는 개념
	- 기본적으로 60초 단위로 수집
	- High resolution 모드에서는 1초 단위
- 이후 초 단위로 1, 5, 10, 30 혹은 60의 배수로 조회 가능

<br>

> 데이터 포인트의 기간(Period)

- 데이터가 얼마만큼의 시간을 기준으로 묶여서 보여지는지에 관한 개념
- 초 단위로 1, 5, 10, 30 혹은 60의 배수
	- 당연히 60초 단위 미만은 High-Resolution 모드의 데이터만 가능
	- 최소 1초에서 최대 86,400초(1일) 까지 묶어서 볼 수 있음
- 보관 기간
	- 60초 미만은 최대 3시간
	- 60초는 15일
	- 300초는 63일
	- 1시간(3600초)는 455일(15개월)
- 작은 단위의 보관 기간은 큰 단위로 계속 합쳐짐
	- 예: 1분 단위는 15일 동안만 확인 가능, 이후 15일이 지나면 5분 단위로만 확인 가능
		- 이후 63일이 지나면 1시간 단위로만 확인 가능
- 주의사항: <font color= pink>2주 이상 데이터가 업데이트 되지 않은 Metric의 경우 콘솔에서 보이지 않음</font>
	- 모든 콘솔에서 사라짐
	- CLI로만 확인 가능

<br>

> 차원(Dimension)

- 일종의 태그/카테고리
	- key-value 로 구성
	- Metric을 구분할 때 사용
- 최대 30개까지 할당 가능
- 예: AWS/EC2 네임스페이스에 모든 EC2의 metric가 수집됨 -> 어떤 인스턴스로부터 metric이 왔는지 확인이 필요
	- InstanceID Dimension 을 통해 EC2 인스턴스 단위로 구분해서 확인
- 조합 가능
	- 예: Dimension : (Server=prod, Domain=Seoul) or (Server=dev)


## END

Reference
- Youtube: [AWS 강의실](https://www.youtube.com/@AWSClassroom)