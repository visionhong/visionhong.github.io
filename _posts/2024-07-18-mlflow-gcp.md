---
title:  "Developing an experimental system for image generation"
folder: "tools"
categories:
  - tools
toc: true
toc_sticky: true
toc_icon: "bars"
toc_label: "목록"
header:
  teaser: "/images/mlflow-gcp/1.png"
---

### Intro

While working in the field of image generation, I encountered a frustrating issue: "How can I easily find the optimal parameters for the model I'm using?"

Given the inherent randomness in image generation, it's challenging to consistently produce desired images. However, I believe that to create a useful service, we need to be able to consistently produce images that meet a certain quality threshold.

To achieve this, I needed to identify which parameters have the most significant impact on performance, based on factual evidence. However, after searching through various communities for methods to find optimal parameters, I couldn't find a clear solution.

Many people share impressive images they've generated in online communities like CivitAI and Reddit, but these are often either lucky outcomes or images created by experienced individuals in the field. While the model, prompt, and parameters used to generate these images are often shared, they can't be universally applied to all situations, making it difficult to generalize.

In other words, image generation results still largely depend on experience, and there's a lack of data-based explanations for why certain results are produced.

Therefore, I wanted to create a tool that could automate image generation, accumulate data, and analyze how images change under different conditions (parameters), and in which cases the results are good or bad.

<br>

### How?

In deep learning, there's a term called hyperparameters. These are top-level parameters (such as epoch, learning rate, batch size, optimizer) that significantly influence the model's learning process. The act of finding the optimal hyperparameter values is called "hyperparameter tuning."

There are many libraries created for hyperparameter tuning (Ray Tune, Optuna, W&B Sweep, etc.). However, what I wanted to do was not adjust hyperparameters for training, but rather 'compare results based on parameter changes.'

Since the results in question are generated images, I needed to be able to easily view images that match specific parameter conditions together. To solve this, I considered two approaches:
 
 
1. Experiment with parameters using W&B's sweep feature and send images to W&B's managed cloud for analysis on the W&B dashboard
   - Pros
     - Easy implementation with a single library
     - Dashboard provided
   - Cons
     - Cost (not open source)
     - Slow rendering speed for larger images, causing inconvenience
2. Use MLflow's optuna integration to experiment with parameters, build an MLflow tracking server to manage artifacts and logs, and develop a separate image viewer UI
   - Pros
     - Low cost (open source)
     - Custom UI
   - Cons
     - Slightly complex architecture setup
     - Requires custom UI development

Ultimately, I chose option 2. It allows for cost-effective operation and the ability to customize the UI according to my desired image generation workflow. The architecture for option 2 is as follows:

![](/images/mlflow-gcp/1.png){: .align-center height="80%" width="80%"}

Typically, when using an MLflow tracking server, hyperparameters and metric values are stored in the backend store, while model files are stored in the artifact store. 

However, I used the backend store to save parameters that influence the image generation inference process and the artifact store to save the generated images.

The MLflow tracking server can be simply installed on an instance for use, or it can be safely operated in a Kubernetes environment like GKE.

However, we need to consider the cost. As it's literally a "server," it requires computing resources, and in a cloud environment, you're billed for what you use.

But do we really need to keep a server running continuously for parameter experiments when we only need it occasionally? I didn't think so, which is why I chose Cloud Run to operate the MLflow tracking server in a serverless manner.

Cloud Run is one of GCP's services that allows you to deploy by running container images in a serverless manner. Similarly, I used GCP's SQL and GCS (Google Cloud Storage) services for the backend store and artifact store, respectively.

Now, let's build the environment step by step.

<br>

### Setting Up MLflow on GCP

> Prerequisite

- Clone [Github Repository](https://github.com/visionhong/mlflow-gcp){:target="_blank" style="color: red;" }
- GCP Account
- GCP CLI
- Docker Environment
  

#### 1. Create Service Account

For the image generation ML code or Image Viewer client to communicate with GCP services, you need a key pair of a Service Account that has the role to access those services.

1. IAM & Admin -> Sevice Accounts -> Create Service Account
   
    ![](/images/mlflow-gcp/2.png){: height="50%" width="50%"}

2. Assign roles to the Service Account (we're setting it as Owner here, but it's recommended to modify it later according to your desired permissions)
   
    ![](/images/mlflow-gcp/3.png){: height="50%" width="50%"}

3. Generate Json key pair

    ![](/images/mlflow-gcp/4.png){: height="80%" width="80%"}

4. Save the key pair
   
    Save the key pair as "credentials.json" and keep it safe. You can only download the key pair once.

<br>

#### 2. Create SQL Database

Create PostgreSQL to be used as the backend store.

1. SQL -> Create Instance -> Select PostgreSQL

    ![](/images/mlflow-gcp/5.png){: height="80%" width="80%"}

2. Name your database, and set the default user password and location preferences

    ![](/images/mlflow-gcp/6.png){: height="50%" width="50%"}

3. Show Configuration options -> set machine and storage

    ![](/images/mlflow-gcp/7.png){: height="50%" width="50%"}

4. Databases -> Create Database

    ![](/images/mlflow-gcp/8.png){: height="80%" width="80%"}

5. Connections -> Security -> Allow only SSL connections

    ![](/images/mlflow-gcp/9.png){: height="50%" width="50%"}

6. Connections -> Security -> Click Create Client Certificate
   
    To allow MLflow to connect to your SQL instance, you need to set up an SSL connection.
When you create a client certificate, you'll download three files: client-key.pem, client-cert.pem, and server-ca.pem. Keep these safe.


    ![](/images/mlflow-gcp/11.png){: height="50%" width="50%"}


7. Users -> Add User Account -> ADD
    
    ![](/images/mlflow-gcp/10.png){: height="80%" width="80%"}

<br>

#### 3. Create Google Storage Bucket

Create a GCS bucket to be used as the artifact store.

1. Cloud Storage -> Buckets -> Create

    ![](/images/mlflow-gcp/12.png){: height="50%" width="50%"}

<br>

#### 4. Secret Manager

The MLflow Tracking server requires several environment variables to be set, such as the URIs for the backend store and artifact store. Since we're building everything in the GCP environment, we'll store these environment variables in GCP's Secret Manager and configure the MLflow Tracking Server to use them securely.

1. Secret Manager -> Create Secret

    We need four Secrets
    1. mlflow_artifact_url: gsutil URI(Cloud Storage -> Click MLflow bucket -> Configutation -> gsutil URI 복사)
        ```
        gs://<bucket name>
        ```
    2. mlflow_database_url: 
        ``` 
        postgresql+psycopg2://<dbuser>:<dbpass>@/<dbname>?host=/cloudsql/<sql instance connection name>
        ```
    3. mlflow_tracking_username: your http auth(login) username
    4. mlflow_tracking_password: your http auth(login) password

<br>

#### 5. Container Registry

Set up the MLflow Tracking Server image to be used in Cloud Run.

1. Create Artifact Registry repo
   
   Artifact Registry -> Create repository

   ![](/images/mlflow-gcp/13.png){: height="50%" width="50%"}

2. Configure code

    Please configure as shown in the image below.
    - Create a secrets folder and move the 4 files you've downloaded so far
    - Modify the environment variables at the top of the Makefile to match your settings. (GCP_PROJECT refers to the project id.)

    ![](/images/mlflow-gcp/14.png){: height="100%" width="100%"}

3. Activate your Google Cloud Service Account (Run in shell)

    ```
    gcloud auth activate-service-account --key-file="<your_credentials_file_path>"
    ```

4. Authenticate Artifact Registry

    ```
    make docker-auth
    ```
5. Docker build & push

    ```
    make build && make tag && make push
    ```

    Check if the image is stored in Artifact Registry as shown in the image below

    ![](/images/mlflow-gcp/15.png){: height="80%" width="80%"}


<br>

#### 6. Cloud Run

Now all preparations for running the MLflow Tracking Server are complete. Let's use Cloud Run to run this image.

1. Cloud Run -> Create Service

    - Select the image you put in Artifact Registry
    - Click Allow unauthenticated invocations (mlflow is protected by HTTP basic authentication)
  
    ![](/images/mlflow-gcp/16.png){: height="60%" width="60%"}

2. Click Container, Volumes, Networking, Security

    - Container -> Settings -> Set Memory 1GB, CPU 2
    - Container -> Variables -> Add `GCP_PROJECT : <GCP project id>`
    - Container -> Cloud SQL connections -> Select the SQL instance you created
    - Security -> Select the Service Account you created earlier
    - After configuration, click Create
  
If the Cloud Run service is created successfully, you'll see a screen like this:

![](/images/mlflow-gcp/17.png){: height="100%" width="100%"}

<br>

#### 7.MLflow login

You can access the MLflow dashboard by clicking the URL in the running Cloud Run service.

When you access the URL, you'll see a login screen like the one below. Enter the `mlflow_tracking_username` and `mlflow_tracking_password` you set in Secret Manager.

![](/images/mlflow-gcp/18.png){: height="50%" width="50%"}

If you see a dashboard like the one below after logging in, congratulations! You've successfully set up the MLflow Tracking Server.

![](/images/mlflow-gcp/19.png){: height="100%" width="100%"}


<br>

### What’s Next?

We've now set up a Serverless MLflow Tracking Server in the GCP environment. In the next post, we'll create a tutorial on how to experiment with parameters using MLflow+Optuna and build a UI that allows you to easily and intuitively analyze the results your image generation model produces.

keep going