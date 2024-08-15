# YOLO Object Detection with PyTorch

This repository contains an implementation of the YOLO (You Only Look Once) object detection algorithm using PyTorch. The project is part of HW2 for the Deep Learning course instructed by Dr. Soleymani.

## Project Overview

YOLO is a real-time object detection system capable of recognizing multiple objects in a single image while predicting their corresponding bounding boxes. This project focuses on implementing YOLO using PyTorch and training the model with the VOC dataset.

## Task: Object Detection with YOLO

The YOLO model is trained to detect and classify objects in images from the VOC dataset. This involves predicting both the object class and the location of bounding boxes for objects within the image. The system is efficient and capable of performing object detection in real-time.

## Dataset

The dataset used is the **Pascal VOC 2012** dataset, which contains images labeled with 20 different object classes, including:
- Aeroplane
- Bicycle
- Bird
- Boat
- Bottle
- Bus
- Car
- Cat
- Chair
- Cow
- Dining table
- Dog
- Horse
- Motorbike
- Person
- Potted plant
- Sheep
- Sofa
- Train
- TV monitor

The dataset is automatically downloaded and extracted within the notebook if it hasn't been previously downloaded.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/AqaPayam/YOLO_Object_Detection_PyTorch.git
    ```

2. Install the necessary dependencies by running the installation commands in the notebook. Required packages include:
    - PyTorch
    - Torchvision
    - Numpy
    - Pandas
    - Matplotlib
    - torchsummary

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook YOLO_Object_Detection.ipynb
    ```

## Running the Model

1. **Dataset Preparation:** The notebook automatically downloads and extracts the Pascal VOC dataset if it is not available.
2. **Training the YOLO Model:** Execute the cells in the notebook to train the YOLO model on the VOC dataset. The model learns to detect objects and predict bounding boxes within images.
3. **Evaluation:** The model's performance is evaluated on test images, displaying predicted bounding boxes along with the object classes.

## Results

The YOLO model can recognize multiple objects in a single image and predict their bounding boxes. The system is optimized for real-time performance.

## Acknowledgments

This project is part of the Deep Learning course by Dr. Soleymani.
