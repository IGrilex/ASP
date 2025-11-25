Document Structure for Presenting SLAM Algorithm Tests

1. Introduction
The purpose of this codebase is a way to test diffrent slam algorithms in order to determin if they are viable for real time use or are just research pappers.

The algorithms tested here are the fallowing:
- ORB SLAM3 
- MAST3R SLAM 
- Air SLAM

2. Test Environment
## 2.1 Hardware Setup (PC Specifications)

### CPU
- **Model:** Intel Core i7-8700K  
- **Clock Speed:** 3.70 GHz (max 3.696 GHz reported)  
- **Cores/Threads:** 6 cores / 12 threads  

### GPU
- **Model:** NVIDIA GeForce RTX 4060 Ti  
- **VRAM:** 8 GB  
- **Driver Version:** 581.57  
- **CUDA Runtime (Driver):** 13.0  

### CUDA Toolkit
- **Installed CUDA Toolkit Version:** 12.4  
- **NVCC Build:** V12.4.99  

### RAM
- **Total:** 16 GB (2 × 8 GB)  
- **Speed:** 3467 MHz  

### Storage
- **Primary Drive:** Samsung SSD 990 PRO  
- **Capacity:** 2 TB  
- **Type:** SSD  

Operating System (version, kernel if Linux)
Nativly my pc runs Windows 10 Pro, but the test were ran using WSL2 with either UBUNTU 22.04 or 20.04

ROS version (if applicable)
ROS Noetic

SLAM-related dependencies

3. Datasets
3.1 Dataset Overview

All the test where ran on the kitti360 dataset (you can find it here https://www.cvlibs.net/datasets/kitti-360/index.php), where i have selected the subsection called "Test SLAM", on this i had to testing scenarios
-"light": that ran only on the test0 from the dataset, this scenario was used for slower algorithms as the inferance time was to big to run on the 
complete dataset
-"full": the complete dataset that contains test0, test1, test2 & test3. Most of the algorithms where tested on this dataset.

Sensor types included 
<for chatgpt to complete from the net>

Resolution and framerate
<for chatgpt to complete from the net>

Duration / number of frames
<for chatgpt to complete from the net, keep in mind we are using just the test SLAM subset>

Challenges in the dataset
(motion blur, dynamic environment, low light, fast motion)
<for chatgpt to complete from the net>

preparation in order to run the tests, from https://www.cvlibs.net/datasets/kitti-360/download.php you need to downlad the fallowing
from the "2D data & labels" section -> Test SLAM (14G)
from the "3D data & labels" section -> Test SLAM (12G)
from the "Calibrations & Poses" section -> Calibrations (3K)
                                        -> Vechicle Poses (8.9M)

after you have extracted the files from the downloaded arhives, you need to place them in a /data folder in the root directory
and after that you need to run python3 ./src/build_kitti360_test_maps.py that will produce the .ply point cloud maps for the GT MAPS.

3.2 Preprocessing Steps

Calibration
<for chatgpt to complete from the net>
Synchronization
<for chatgpt to complete from the net>
Any downsampling, noise filtering, or format conversions
no downsampling or noise filtering was done, we ran the dataset as it was, the point of the test is to see how the models perform out of the box

4. SLAM Algorithms Tested

Create one subsection per algorithm.

4.1 Python ORB SLAM3
You can find the github for this project here https://github.com/xingruiyang/ORB-SLAM3-python
I chose to run the python rapped algorithm in order to keep the complete codebase in python and also because i am a little rusty at C++. So keeping this
in mind, all the resoults we get from ORB are underestimation on what the optimal performance is.

ORB SLAM3 was teste in both mono and stereo
the runners are called in the fallowing way:
-> python3 ./src/run_mono_ORBSlam3.py
-> python3 ./src/run_stereo_ORBSlam3.py

Setup problems:
With ORB SLAM i didnt have any problems setting it up, i just created a WSL2 Ubuntu 22.04 and in there i cloned the github from above and build it, it ran
smooth, fast and preaty decent accuracy.

4.2 MAST3R SLAM
You can find the github for this project here https://github.com/rmurai0610/MASt3R-SLAM
Now for this algorithm i had a lot of problems and it was a pain to get it running. First i have tried to fallow the github instruction but it would 
always fail the building as there where dependencies errors. In order to get it running i have fallowed this youtube video https://www.youtube.com/watch?v=TK8DK19o6YQ
where it shows you step by step on how to run it. I also had problems building it so the only advice i can give you is run the steps from the video but
each time it says "conda install ...." use "pip install ...." as this was the only way i got it running.

The algorithm predicts a dense point cloud map, but is atrociacly slow, in hopes to get it running faster, i have changed the MAST3R slam internal files
and added autocast(fp16) for running the MAST3R decoder. i also have reduce the number of frames that are inputed into the algorithm, from each frame to only the only the
3rd frame (you will see some helper functions to corectly alighn the poses timestamps). It improved the runnig speed but it was still SUPER slow.

4.3 AIR SLAM:
You can find the github for this project here https://github.com/sair-lab/AirSLAM
For this i had to make a new linux OS as the ROS Neoletics isnt supported on the Ubuntu 22.04. First i have tried to run it Nativly on the ubuntu 20.04 but
failed as no matter what i did it wouldt build. There for i went with the recommneded path from the github witch is WSL2 Ubuntu 20.04 + Desktop Docker, this worked well and i didnt need to change any thing in the internal
files in order to get it running.

when making the docker i also coppied the code base (and the dataset) in it with the Air Slam git clone and called the container "air_slam". i run the fallowing commands everry time i want to enter in the container:
1. docker start air_slam
2. docker exec -it air_slam /bin/bash (verry important)
3. cd /workspace
4. source devel/setup.bash
5. cd /ASP

in order to make it run you need to run the fallowing script First:
python3 ./src/make_data_4_AirSlam.py
this will rearange the input images in the aspected way for AIR SLAM.

After this you will have the new database and then you can test the algorithm with the command:
python3 ./src/run_AirSlam.py

I couldnt get this to be imported in the code so it was a walkarround where i call the script from inside the code.

Detected problems:
I couldnt run the algorithm with a given script (so just inport it and run it by inserting picture by picture)
the output map can be viewed or converted .ply