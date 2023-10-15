## SNEAKER REVIEW
# Front and Input Page:
![Screenshot (102)](https://github.com/stxrboi/202003490-sneaker-api/assets/73634482/e2c67e65-8cf9-451d-bf08-5f509452e074)
# Result:
![Screenshot (103)](https://github.com/stxrboi/202003490-sneaker-api/assets/73634482/5373a151-8517-4b8f-9ee9-051457596fba)
# Introduction
Our latest sneaker has hit the shelves and we would love to hear from you. The Sneaker Review tool is an application built to give you the consumer the mic and have your views felt by leaving a review of the sneaker. Fully built in Django, utilising the Django Rest Framework for a RESTful API. 

# Dataset
(Description of Dataset)
- Name: sneakers_Reviews_Dataset.csv
  
The dataset has the following features:
- review_id, product_id, user_id: instance of review, product being reviewed, user reviewing.
- rating: 1-5, with 1-2 being negative, 3 neutral and 4-5 positive.
- review_text: the reviews made on the sneaker.
- timestamp: time review was made.

An additional column was created to convert ratings to Sentiments (Negative, Neutral, Positive)

# Features
## Backend w/ Django RestFramework API
- #### Classification Endpoint:
Users enter review and get back sentiment.
- #### Dockerized:
Application is dockerized enabling ease of use, scaling and deployment.
- #### Frontend:
A slick, User-Friendly interface for quick and easy sentiment analysis.
- #### AWS Integration and Deployment:
The Sneaker Review Tool is deployed and hosted on AWS through the ECR, EWS, FARGATE & LOAD BALANCER.
- #### CI/CD:
GitHub actions is implemented enabling docker image to be built and pushed to AWS on push demand.

# LOCAL USE
To make use of this project and implement it locally you must:
* have Docker Desktop installed.
* clone this repo.
* build docker image.
* run docker image.

##### Aditional Resources used:
* https://www.youtube.com/watch?v=t-9lWdZcrQM&t=249s&ab_channel=IntegrationNinjas
* lecture recordings
* https://www.youtube.com/watch?v=o7s-eigrMAI&ab_channel=BeABetterDev
