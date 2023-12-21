
# Playing Cards Image Classification

## ML project for ML Zoomcamp 2023 capstone 1

In this project I have build a Classifier model for image classification of playing cards

This project is based on this dataset:

[Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

(I have also found this [Card Classification](https://www.kaggle.com/datasets/gauravduttakiit/card-classification) dataset, and manually compared them. All the images from it is in the chosen dataset, and also the chosen have 7624 train images, and this other only 4776. I randomly select several images from it and found them in the bigger one. In rare cases quality in the smaller was better, but not to some great extent. )


This dataset have 53 classes 7624 train, 265 test, 265 validation images 224 X 224 X 3 jpg format. 

All the work with data, selecting best model, traings is presented in the notebook.ipynb.

*In this project a CNN model is developed, trained and deployed. This model is doing card image classfications
I have reached a 0.913 validaion accuracy*


## Web interface

This project is deployed with flask service running in Google Cloud. It is accesible through web interface.
So you can upload your image to the form in the following link to see the model output.


[CARD CLASSIFICATION web page](https://cardclass-2avfrxfgrq-lm.a.run.app/)



## Project files:

**dataset/** - dataset with 'train/' 'valid/' and 'test/' subdirs
in this repository subfolders is empty, but cards_dataset.zip is presented with all images, and folder structure

**notebook_full.ipynb** - Jupiter notebook of preparation, exploration of the dataset, training different models, and comparing them. Full notebook - very long, with my trial and error path

**notebook.ipynb** - Jupiter notebook starting when I reach 0.8 accuracy. Then there is a lot of tuning of hyperparamets. 

**train.py** - script for training the model, using selected best approach and hyperparameters found in the notebook

**cardmodel_v2_40_0.913.h5** - .h5 model file with best validation accuracy, I have got during the training

**convert.py** - script to convert .h5 into .tf-lite model file

**model/cardmodel.tflite** - the best model in tf-lite fromat. 

**cardclass_simple.py** - script for simple prediction with url and local_file function classification, using tf-lite

**cardclass.py** - script for running flask server, with API JSON classification, and classification throug web interface

**test_api_cardclass.py** - python script to test the API

**Dockerfile** - a docker file for building the image.

**Pipfile**
**Pipfile.lock** - files for Pipenv, used in builing the container



For building image and deployment I have used:

**docker build -t cardclass .**

then i run this container with:

**docker run -it -p 9696:9696 cardclass:latest**

then I log in to GPC (Google Cloud Project Console) from local terminal, 
created a GCP image repository, enabled GCP specific API for working with
images, added a tag to my image, pushed it there, and run this container via Cloud Run service.

commands:
**gcloud auth login
gcloud config set project YOUR_PROJECT_ID
docker tag cardclass:latest gcr.io/YOUR_PROJECT_ID/cardclass:latest
gcloud auth configure-docker
docker push gcr.io/YOUR_PROJECT_ID/cardclass:latest**

[CARD CLASSIFICATION web page](https://cardclass-2avfrxfgrq-lm.a.run.app/)

Or, to use with API, it can be accessed withthis url: 
[prediction API](https://cardclass-2avfrxfgrq-lm.a.run.app/predict)



