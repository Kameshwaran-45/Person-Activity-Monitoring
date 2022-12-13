# Person-Activity-Monitoring

Monitoring and tracking the activiy of a person using Pose Estimation and StrongSORT. Customization can be made based on the business requirement.

## Pose Estimation

HRNet with Top down approach is used for this application, where the network is built for estimating keypoints based on person bounding boxes which are detected by another network (yolov5m)

## StrongSORT

StrongSORT tracking algorithm, the upgradation of classic tracker DeepSORT from various aspects, i.e., detection, embedding and association. 

## Quick start

To run this project

1. cloning the repo

```bash
  git clone https://github.com/Kameshwaran-45/Person-Activity-Monitoring.git
```

2. change the directory

```bash
  cd Person-Activity-Monitoring/
```

3. Installing the pre-requisites

```bash
  pip install -r requirements.txt
```

4. Move the downloaded model file inside the directory and set the config.json and run main.py

```bash
  python detect.py
```
