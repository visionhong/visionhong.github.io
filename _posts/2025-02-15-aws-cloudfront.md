---
title:  "Deep dive into AWS CloudFront"
folder: "aws"
categories:
  - aws
header:
  teaser: "/images/aws-cloudfront/architecture.png"
enable_language_selector: true
ko_toc:
  - id: "intro"
    title: "Intro"
  - id: "what-is-cloudfront"
    title: "What is CloudFront?"
  - id: "tutorial"
    title: "Tutorial"
  - id: "cloudfront-key-points"
    title: "CloudFront Key Points"
  - id: "conclusion"
    title: "Conclusion"
en_toc:
  - id: "intro-1"
    title: "Intro"
  - id: "what-is-cloudfront-1"
    title: "What is CloudFront?"
  - id: "tutorial-1"
    title: "Tutorial"
  - id: "cloudfront-key-points-1"
    title: "CloudFront Key Points"
  - id: "conclusion-1"
    title: "Conclusion"
toc_label: "Contents"
---

{% include language-selector.html %}

<div id="korean-content" class="language-content" markdown="1">

### Intro

이미지를 생성을 위한 글로벌 서비스를 운영할때 효율적인 이미지 전송은 중요한 과제 중 하나입니다. 사용자가 이미지를 요청할 때마다 발생하는 지연 시간을 최소화하고, 전 세계 어디서든 안정적으로 제공하려면 CDN(Content Delivery Network)의 도움이 필수적입니다.

AWS의 CloudFront는 이러한 요구 사항을 만족시키는 대표적인 CDN 서비스로, S3에 저장된 정적 콘텐츠(이미지, 동영상 등)를 전 세계 에지 로케이션에서 빠르고 안전하게 전송해 줍니다.

이번 포스트에서는 이미지 생성 → S3 업로드 → CloudFront 전송으로 이어지는 전 과정을 단계별로 살펴보며, 어떻게 하면 생성된 이미지를 최적화된 속도로 전 세계 사용자에게 제공할 수 있는지 구체적으로 알아보겠습니다.

<br>

### What is CloudFront?

AWS CloudFront는 전 세계 Edge Location에 캐시를 두고 사용자가 웹사이트, 애플리케이션, 또는 기타 콘텐츠(이미지, 동영상, HTML, CSS, JavaScript 등)에 빠르게 액세스할 수 있도록 돕는 CDN(Content Delivery Network)입니다.


> 왜 CloudFront를 사용해야 할까?

- 글로벌 캐시: 전 세계 여러 지역에 캐시 서버를 배포해, 사용자와 물리적으로 가까운 에지 로케이션에서 콘텐츠를 제공함으로써 지연(Latency)을 획기적으로 줄일 수 있습니다.
- 보안: AWS WAF(Web Application Firewall) 또는 AWS Shield(DDos 방어)와 연동해 보안성을 강화할 수 있습니다. 또한 HTTPS 지원 및 인증서 설정을 통해 트래픽을 안전하게 전송할 수 있습니다.
- 비용 절감: Origin(ex. S3 Bucket)의 트래픽을 줄여 비용을 절감하고, CloudFront의 캐시 정책을 활용해 효율적으로 트래픽을 제어할 수 있습니다.
- 유연한 설정: 캐시 정책, 동적 콘텐츠, Lambda@Edge 등 다양한 기능을 활용하여 맞춤화된 CDN 환경을 구성할 수 있습니다.

<br>

### Tutorial

> Architecture

![](/images/aws-cloudfront/architecture.png){: .align-center height="80%" width="80%"}

1. 유저가 이미지 생성을 요청
2. 모델 서버가 이미지를 생성
3. 생성된 이미지를 Amazon S3 Bucket에 업로드
4. CloudFront는 원본(origin)으로 S3 버킷을 설정하여, 클라이언트(사용자)가 요청하면 전 세계 에지 로케이션에서 이미지를 서빙
5. 클라이언트는 https://{CloudFront 도메인}/path/to/image.png 으로 접근

<br>

> S3 Bucket Settings

- S3에서 버킷을 생성할때 특정 리전을 선택해도 CloudFront는 전 세계 에지 서버에서 빠르게 제공 가능 합니다.
 ![](/images/aws-cloudfront/s3-region.png){: .align-center height="100%" width="100%"}

- CloudFront를 통해 이미지를 서빙하려면, 기본적으로 S3 버킷은 퍼블릭 액세스를 차단하는 것이 권장됩니다. 대신, Origin Access Control(OAC) 또는 Origin Access Identity(OAI)를 사용하여 S3 원본에 접근할 수 있도록 설정할 수 있습니다.(퍼블릭 읽기 권한을 부여해도 동작은 하지만, 보안 측면에서는 OAC/OAI 설정이 더 안전합니다.)
  ![](/images/aws-cloudfront/s3-public-access.png){: .align-center height="100%" width="100%"}

- 이미지 업로드 및 경로 관리
생성된 이미지를 S3://my-image-bucket-2025/outputs/2025-02-01/image_001.png 와 같은 경로에 업로드한다고 가정합니다.
이미지를 CloudFront로 서빙할 때는 S3 버킷 내의 객체 키(object key)가 그대로 사용됩니다.
- ex) S3의 객체 키가 outputs/2025-02-01/image_001.png라면, CloudFront 도메인은 https://{CloudFront 도메인}/outputs/2025-02-01/image_001.png 형태로 접근 가능

<br>

> CloudFront Distribution

Origin Settings
- Origin Domain: 배포하고자 하는 S3 버킷을 선택합니다.(my-image-bucket-2025.s3.amazonaws.com)
- Origin Access: **Origin Access Control(OAC)**을 사용하는 경우, “Create control setting” 등을 통해 설정할 수 있습니다.
- OAI를 사용하는 경우 “Origin Access Identity(OAI)”를 생성 또는 기존 OAI 사용을 선택 후 “Update Bucket Policy” 버튼을 통해 자동으로 버킷 정책을 설정해줄 수 있습니다.
  ![](/images/aws-cloudfront/cloudfront-origin.png){: .align-center height="100%" width="100%"}

<br>

Cache Policy
- Cache Key and Origin Requests:
  - CloudFront에는 **캐시 정책(Cache Policy)**이라는 개념이 있습니다.
  - 기본적으로 Cache Policy: CachingOptimized 또는 CachingDisabled 등을 선택할 수 있으며, 이미지를 캐싱하려면 CachingOptimized 또는 커스텀 정책을 사용합니다.
  ![](/images/aws-cloudfront/cloudfront-cache.png){: .align-center height="100%" width="100%"}

- Object Caching:
  - S3 객체에 Cache-Control 헤더를 지정하지 않았다면, CloudFront 디폴트 TTL 값을 사용할 수 있습니다.
  - 이미지는 보통 변경 주기가 길기 때문에 충분히 긴 max-age 값을 사용할 수 있습니다. 다만, 이미지가 자주 변경된다면 재배포 정책을 수립하거나 버전 관리 전략(파일명에 버전 문자열 포함 등)을 적용하는 것이 중요합니다.

- Compress Objects Automatically:
  - 텍스트 파일(HTML, CSS, JS 등)의 경우 GZIP 혹은 Brotli 압축 전송을 지원하도록 체크합니다. 이미지는 일반적으로 JPEG, PNG, WebP 등 자체 압축 포맷이 있으므로 별도 설정이 필요 없지만, 텍스트 리소스가 있다면 이 옵션을 활성화합니다.

<br>

> Update S3 Bucket Policy

- 위 설정을 완료한 후 “Create Distribution”을 클릭하면, CloudFront에서 **배포(Distribution)**를 생성합니다.
- 새로운 OAC를 생성한 경우 아래 이미지처럼 S3 bucket podicy를 업데이트 하라는 안내가 나옵니다. 해당 policy를 복사하여 Bucket -> Permissions -> Bucket policy에 정책을 추가합니다.
  ![](/images/aws-cloudfront/cloudfront-policy.png){: .align-center height="100%" width="100%"}

  ``` json
  {
        "Version": "2008-10-17",
        "Id": "PolicyForCloudFrontPrivateContent",
        "Statement": [
            {
                "Sid": "AllowCloudFrontServicePrincipal",
                "Effect": "Allow",
                "Principal": {
                    "Service": "cloudfront.amazonaws.com"
                },
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::my-image-bucket-2025/*",
                "Condition": {
                    "StringEquals": {
                      "AWS:SourceArn": "arn:aws:cloudfront::079160317096:distribution/EU1T2QP4ZW3MM"
                    }
                }
            }
        ]
      }
  ```

<br>

### CloudFront Key Points

> Cache

클라이언트가 'https://d111111abcdef8.cloudfront.net/outputs/2025-02-01/image_001.png'를 요청하면, CloudFront는 다음 절차로 응답합니다:

1. 에지 로케이션 캐시에 해당 객체가 있는지 확인.
2. 없다면 원본(S3)으로 요청 → 응답을 가져와서 에지 로케이션에 캐싱.
3. 다음 요청부터는 캐시 유효 기간 내에서는 에지 로케이션에서 바로 응답.
  
<br>

> Invalidation

이미지가 업데이트되거나, 동일한 키(예: image_001.png)로 다시 업로드된다면 다음과 같은 문제를 겪을 수 있습니다.
- **캐시 무효화(Invalidation)**: 기존 캐시에 있는 이미지가 그대로 서빙될 수 있어, 사용자가 새로운 이미지를 보지 못할 수 있습니다. 이때 CloudFront 콘솔이나 CLI를 통해 Invalidate 처리를 해주면 됩니다.
  ![](/images/aws-cloudfront/cloudfront-invalidation.png){: .align-center height="100%" width="100%"}
- CloudFront 콘솔에서 해당 Distribution을 선택한 후, 상단 메뉴의 “Invalidations” > “Create Invalidation”를 클릭합니다.
경로로 outputs/2025-02-01/image_001.png 또는 /* 등을 지정해 캐시를 무효화합니다.
무효화 요청 후 수 분 이내에 에지 로케이션에서 해당 객체가 제거되어, 새로운 버전이 요청 시점에 다시 캐싱됩니다.
  ![](/images/aws-cloudfront/cloudfront-invalidation2.png){: .align-center height="100%" width="100%"}
  
  
- 버전 관리 전략: 이미지 파일 이름에 버전 정보를 붙여(image_001_v2.png) 변경될 때마다 새로운 키를 사용하면, 추가 인벌리데이션(Invalidation) 없이도 새 객체를 바로 제공할 수 있습니다.

<br>


> URL

- URL 접근
  - CloudFront 생성 후 제공되는 도메인 이름을 통해 이미지를 접근할 수 있습니다.
예: https://d111111abcdef8.cloudfront.net/outputs/2025-02-01/image_001.png
- Alternate domain names (선택 사항)
  - CloudFront를 통해 https://images.mycompany.com/outputs/2025-02-01/image_001.png 처럼 사용자 정의 도메인을 쓸 수도 있습니다.
  - Route 53 혹은 외부 DNS에서 CNAME 설정을 통해, images.mycompany.com이 CloudFront의 도메인(d111111abcdef8.cloudfront.net)을 가리키도록 합니다.
  - CloudFront 배포 설정 > Alternate Domain Names(CNAME) 항목에 images.mycompany.com을 추가합니다.
  - **AWS Certificate Manager(ACM)**를 통해 images.mycompany.com에 대한 인증서(SSL/TLS) 발급 후, 해당 인증서를 배포에 연결합니다.
  - 이렇게 하면 사용자들은 CloudFront 기본 도메인 대신 images.mycompany.com으로 환경변수로 관리하기 편한 URL을 사용할 수 있습니다.

<br>

### Conclusion

AWS CloudFront를 사용하면 S3에 저장된 이미지 파일을 전 세계적으로 빠르고 안정적으로 서빙할 수 있습니다. 주요 이점은 다음과 같습니다.

- 전 세계 에지 로케이션에서 콘텐츠를 제공해 지연을 최소화합니다.
- 캐싱을 통해 S3 트래픽 및 비용 절감 효과가 있습니다.
- 보안 설정(OAC, OAI, WAF, HTTPS)을 통해 안전하게 이미지 등을 제공할 수 있습니다.
- 이미지가 업데이트되는 경우 버전 관리 또는 Invalidation을 통해 캐시 동작을 제어할 수 있습니다.
- 필요에 따라 Lambda@Edge 등의 기능을 통해 동적 처리를 적용할 수도 있습니다.

이미지 생성 파이프라인(예: Lambda, EC2에서 이미지를 생성 후 S3에 업로드)과 연결하여, 최종 사용자에게 신속하게 이미지를 제공하려면 CloudFront를 적극 활용하는 것이 좋습니다.

CloudFront는 단순 정적 콘텐츠 캐싱을 넘어, 다양한 시나리오(동적 콘텐츠, 서드파티 통합, 보안 강화 등)에 대응할 수 있는 강력하고 유연한 CDN 서비스이므로, 규모가 커지거나 전 세계 사용자 대상의 서비스에서 필수적으로 고려되는 구성 요소입니다.

</div>

<div id="english-content" class="language-content" style="display: none;" markdown="1">


### Intro

When operating a global service for image generation, efficient image delivery is one of the key challenges. Minimizing latency whenever a user requests an image and providing reliable service from anywhere in the world becomes much easier with the help of a CDN (Content Delivery Network).

AWS CloudFront is a representative CDN service that meets these needs, rapidly and securely delivering static content (images, videos, etc.) stored in Amazon S3 from edge locations around the world.

In this post, we’ll walk step-by-step through the entire process—image generation → S3 upload → CloudFront delivery—to see how to provide generated images at optimized speed to users worldwide.

<br>

### What is CloudFront?

AWS CloudFront is a CDN (Content Delivery Network) that places caches in edge locations around the globe, allowing users to quickly access websites, applications, or other content (images, videos, HTML, CSS, JavaScript, etc.).


> Why use CloudFront?

- Global Cache: Content is delivered from an edge location physically close to the user through multiple cache servers deployed worldwide, drastically reducing latency.
- Security: You can reinforce security by integrating with AWS WAF (Web Application Firewall) or AWS Shield (DDos protection). It also supports HTTPS and certificate settings to securely transmit traffic.
- Cost Reduction: It reduces traffic to the origin (e.g., S3 Bucket) to save costs, and you can manage traffic efficiently by utilizing CloudFront cache policies.
- Flexible Configuration: You can create a customized CDN environment by using various features like cache policies, dynamic content, Lambda@Edge, and more.

<br>

### Tutorial

> Architecture

![](/images/aws-cloudfront/architecture.png){: .align-center height="80%" width="80%"}

1. The user requests an image generation
2. The model server generates the image
3. The generated image is uploaded to Amazon S3 Bucket
4. CloudFront is configured with the S3 bucket as the origin. When the client (user) requests the image, CloudFront serves it from edge locations around the world
5. The client accesses it via https://{CloudFront Domain}/path/to/image.png

<br>

> S3 Bucket Settings

- Even if you select a specific region when creating an S3 bucket, CloudFront can still deliver content quickly from edge servers worldwide.
 ![](/images/aws-cloudfront/s3-region.png){: .align-center height="100%" width="100%"}

- By default, it is recommended that your S3 bucket block public access when serving images through CloudFront. Instead, you can configure Origin Access Control (OAC) or Origin Access Identity (OAI) to allow CloudFront to access the S3 origin. (It will still work if you grant public read permission, but in terms of security, using OAC/OAI is safer.)
  ![](/images/aws-cloudfront/s3-public-access.png){: .align-center height="100%" width="100%"}

- Image upload and path management  
  Let’s assume you upload generated images to a path like S3://my-image-bucket-2025/outputs/2025-02-01/image_001.png.  
  When serving the image through CloudFront, the S3 object key is used as is.
  - For example, if the object key in S3 is outputs/2025-02-01/image_001.png, you can access it via https://{CloudFront Domain}/outputs/2025-02-01/image_001.png through CloudFront.

<br>

> CloudFront Distribution

**Origin Settings**
- **Origin Domain**: Select the S3 bucket you want to distribute (my-image-bucket-2025.s3.amazonaws.com).
- **Origin Access**: If you’re using **Origin Access Control (OAC)**, you can configure it via “Create control setting” and so on.
- If using OAI, select “Origin Access Identity (OAI)” (create new or use existing) and then click “Update Bucket Policy” to automatically set up the bucket policy.
  ![](/images/aws-cloudfront/cloudfront-origin.png){: .align-center height="100%" width="100%"}

<br>

**Cache Policy**
- **Cache Key and Origin Requests**:
  - CloudFront has the concept of a **Cache Policy**.
  - By default, you can select from Cache Policy options like `CachingOptimized` or `CachingDisabled`. To cache images, use `CachingOptimized` or a custom policy.
  ![](/images/aws-cloudfront/cloudfront-cache.png){: .align-center height="100%" width="100%"}

- **Object Caching**:
  - If your S3 objects do not have a `Cache-Control` header set, you can use the default CloudFront TTL.
  - Images usually have a long update cycle, so you can use a sufficiently long `max-age`. However, if images change frequently, it’s important to establish a redeployment policy or versioning strategy (e.g., include version strings in filenames).

- **Compress Objects Automatically**:
  - For text files (HTML, CSS, JS, etc.), enable GZIP or Brotli compression by checking this option. Images typically use their own compression formats (JPEG, PNG, WebP), so no separate configuration is necessary. But do enable this for text resources if applicable.

<br>

> Update S3 Bucket Policy

- After completing the above settings, click **“Create Distribution”** to create the CloudFront **Distribution**.
- If you create a new OAC, you may see a message prompting you to update the S3 bucket policy, as shown in the image below. Copy this policy and add it to the Bucket -> Permissions -> Bucket policy.
  ![](/images/aws-cloudfront/cloudfront-policy.png){: .align-center height="100%" width="100%"}

  ```json
  {
    "Version": "2008-10-17",
    "Id": "PolicyForCloudFrontPrivateContent",
    "Statement": [
      {
        "Sid": "AllowCloudFrontServicePrincipal",
        "Effect": "Allow",
        "Principal": {
          "Service": "cloudfront.amazonaws.com"
        },
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-image-bucket-2025/*",
        "Condition": {
          "StringEquals": {
            "AWS:SourceArn": "arn:aws:cloudfront::079160317096:distribution/EU1T2QP4ZW3MM"
          }
        }
      }
    ]
  }
  ```

<br>

### CloudFront Key Points

> Cache

When a client requests ‘https://d111111abcdef8.cloudfront.net/outputs/2025-02-01/image_001.png’, CloudFront responds as follows:

1. The edge location checks whether it has the requested object in its cache.
2. If not, it requests it from the origin (S3), retrieves the response, and caches it at the edge location.
3. Future requests within the cache validity period are served directly from the edge location’s cache.

<br>

> Invalidation

When an image is updated or the same key (e.g., image_001.png) is used for re-upload, the following issues can occur:
- **Cache Invalidation**: If the previous version of the image remains in the cache, users may not see the updated image. In this case, you can invalidate the cache via the CloudFront console or CLI.
  ![](/images/aws-cloudfront/cloudfront-invalidation.png){: .align-center height="100%" width="100%"}

- In the CloudFront console, select the Distribution, then in the top menu choose “Invalidations” > “Create Invalidation.”  
  Enter the path such as `outputs/2025-02-01/image_001.png` or `/*` to invalidate.  
  After a few minutes, the object is removed from edge caches, and the new version is fetched and cached upon the next request.
  ![](/images/aws-cloudfront/cloudfront-invalidation2.png){: .align-center height="100%" width="100%"}

- **Versioning Strategy**: By adding version information to the filename (e.g., image_001_v2.png), you can serve the new object without needing additional invalidations.

<br>

> URL

- **URL Access**
  - After CloudFront is created, you can access the image through the domain name provided by CloudFront.  
    Example: `https://d111111abcdef8.cloudfront.net/outputs/2025-02-01/image_001.png`
- **Alternate domain names (optional)**
  - You can use a custom domain with CloudFront, such as `https://images.mycompany.com/outputs/2025-02-01/image_001.png`.
  - Configure your DNS (Route 53 or external) by creating a CNAME record so that `images.mycompany.com` points to the CloudFront domain (`d111111abcdef8.cloudfront.net`).
  - Add `images.mycompany.com` to the **Alternate Domain Names (CNAME)** field in your CloudFront distribution settings.
  - Use **AWS Certificate Manager (ACM)** to issue an SSL/TLS certificate for `images.mycompany.com` and attach it to the distribution.
  - This way, users can use a convenient URL managed via environment variables instead of the default CloudFront domain.

<br>

### Conclusion

By using AWS CloudFront, you can serve image files stored in S3 quickly and reliably to users around the world. The main benefits are:

- Minimize latency by delivering content from edge locations around the globe.
- Reduce S3 traffic and costs through caching.
- Securely deliver images and other content with features like OAC, OAI, WAF, and HTTPS.
- Control caching behavior through versioning or invalidation if images are updated.
- Optionally leverage Lambda@Edge and other functionalities for dynamic processing.

CloudFront goes beyond simple static content caching and offers powerful and flexible CDN services capable of handling various scenarios (dynamic content, third-party integration, advanced security, etc.). It’s a must-have component for services at scale or those targeting a global audience, especially when quickly delivering newly generated images to end users.

</div>