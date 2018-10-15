## Packaged Python Algorithm for classifying sentiment from tweets

----

The aim of this project is to serve as a simple example of implementing keras model (with tensorflow backend) to be used with the [sagemaker-pipeline](https://github.com/MustafaWaheed91/sagemaker-pipeline) project.

### Prerequisites

1. Make sure to have setuptools library installed on python

2. Using version 3.6+ of python

3. Download and unzip [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140) from Kaggle under *./kr-classifytweet/classifytweet/data/* for this python package.

### Running Model locally

```

git clone https://github.com/MustafaWaheed91/kr-classifytweet.git

cd kr-classifytweet

pip3 install -e .

python3 classifytweet/train.py

```

### Running Model on Amazon SageMaker

Follow the instructions in [sagemaker-pipeline](https://github.com/MustafaWaheed91/sagemaker-pipeline) project to see how to use the model
in this package to run with SageMaker training pipeline.

----

### Built primarily with

* [Keras](https://keras.io/) - Python Deep Learning Library


### Authors

* **Mustafa Waheed** - *Data Scientist* 
