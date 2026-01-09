# Project Description: Cifake Classification
A reproducible cifake classification model in a MLops dynamic framework

In this project, we will try to classify AI-generated images from human-generated images in the field of art. We will be using the dataset called “CIFAKE: Real and AI-Generated Synthetic Images”. The dataset consists of the CIFAR10-dataset, a dataset consisting of 60000 coloured 32x32 images, labeled to 10 classes, and 60.000 synthetically generated images of the CIFAR10 classes. The dataset is split into 50.000 training-images and 10.000 testing-images per class. The model used for image generation was Stable Diffusion, version 1.4. Although the dataset is based on the CIFAR10 dataset, we are not interested in what the object on the image is, but whether it is real or fake. Therefore, we do not label the images to 10 different classes but rather labelled as either “REAL”- meaning human generated or “FAKE”- meaning AI-generated. The dataset has a total size of 110 MB and is publicly available on Kaggle. 

We intend to use convolutional neural networks (CNN), implemented through the deep-learning framework Pytorch. We use CNN’s due to their efficiency in image-classification. Furthermore, we will use both Pytorch-Lightning to eliminate the boilerplate code for training and Torchvision to transform our datasets. 
We will use Weights-and-Biases to try different configurations of hyperparameters and architectures and use their built-in Bayesian Optimization option to go for the best-possible model. The results of our experiments will also be logged here.
The first few days will be spent setting up a reproducable virtual environment with  docker, cookiecutter and uv. Then the data-processing, the model, as well as the training and validation processes, will all be implemented with the help of debugging tools, and finally we will format everything to fit PEP8-guidelines.

Our overall goal is to train a model that outperforms random guessing. Maybe, if time permits it, we will also try to compare our final model to the VGG16- a deep convolutional neural network. Another goal for us is to gain a better understanding of the MLOPs-tools taught in the course, and to be able to build a project with perfectly reproduceable results. 

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

