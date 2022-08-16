# How to execute
To execute different experiments on the following commands
can be executed.

With Smaller Datasets.
Experiment 1:-
Without Random zoom, only has random rotations.

python train.py --epochs 10 --duplication_factor 3 --experiment 0

Experiment 2:-
With both random rotation and random zoom 

python train.py --epochs 10 --duplication_factor 3 --experiment 1

With Larger Datasets
Experiment 1:-
Without Random zoom, only has random rotations.

python train.py --epochs 50 --duplication_factor 10 --experiment 0

Experiment 2:-
With both random rotation and random zoom 

python train.py --epochs 50 --duplication_factor 10 --experiment 1



# About the project

The project uses a similar architecture as DETR. However changes where made to the project to include the major available datasets for grasp detection mainly the Cornell dataset for grasping and the jacquard Dataset. Another major change was how to find grasp boxes as well as their orientation. The loss was calculated based on IOU using hungarian matcher.

![](./tensor([150]).png)
![](./figures/img_3dbbox_000134.png)



