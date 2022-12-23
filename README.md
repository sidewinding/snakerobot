# The Influence of Non-zero Slope on Snake Robot Sidewinding Locomotion in Granular Terrains
Created by Lei Huang, Hengqiang Ming, and Yuehong Yin

## Introduction
This work is based on our T-Mech paper: The Influence of Non-zero Slope on Snake Robot Sidewinding Locomotion in Granular Terrains. The article reveals the biological mechanism of snake sidewinding locomotion in granular terrain, and the proposed snake robot has potential applications in desert exploration and rescue.

## Dependencies
### Requirements:
- Taichi Lang (Taichi: High-Performance Computation on Sparse Data Structures) https://github.com/taichi-dev/taichi
- Matlab 2022

The experimental process is illustrated in the video folder:
[Experimental video](https://github.com/sidewinding/snakerobot/blob/main/video/Supplementary%20Movie%20S1.mp4).

https://user-images.githubusercontent.com/97268100/209332671-2e54ebf7-8576-47dc-9177-e71f859de653.mp4
![image](https://github.com/sidewinding/snakerobot/blob/main/picture/1.png)

The 3D printing model of machine snake is illustrated in folder: /3D printing model.
The DRFM model is illustrated in folder: /DRFM model.
The MPM model is illustrated in folder: /MPM model.

## Performance

The accuracy of proposed MPM is similar to the SPH method proposed by Askari et al., but ours is faster.

![image](https://github.com/sidewinding/snakerobot/blob/main/picture/mpm.png)

Some demo:
[intrusion video](https://github.com/sidewinding/snakerobot/blob/main/video/intrusion.mp4).
[sand video](https://github.com/sidewinding/snakerobot/blob/main/video/sand.mp4).
[sand_water video](https://github.com/sidewinding/snakerobot/blob/main/video/sand_water.mp4).

## Usage
To used DRFM demo
```
cd /DRFM model

using Matlab open and run snakemodel.m for Zero slope angle sidewinding.
using Matlab open and run snakemodel15.m for Non-zero slope angle sidewinding.
```
Then the result will be shown in a figure.

To use MPM demo

```
cd /MPM model
Python intrusion.py
```
Then the result will be shown in a GUI.
