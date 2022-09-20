## Basic Action Recognition Project
# Utilizes MMAction Library from Open-MM
Link: https://github.com/open-mmlab/mmaction2

The main reason that I chose to use this library in particular is that it has a plethora of models available for testing and is also able to accept more. With the short time available for this project, I thought it was best to achieve the Minimum Viable Product, rather than completely build from scratch. This library has several models, including transformer models, which are pre-trained for action recognition. The process we use is based upon the SpatioTemporal Action Detection Webcam Demo. 

This utilizes a 3D CNN accepting 8 frames of input at a given time. SpatioTemporal Models have been shown to be exceptional within this problem set. However, this comes at somewhat of a cost in real-time applications. While this project still runs at real-time, it is very choppy as it must wait for model output before drawing. Originally, I had planned to multithread this, similar to the demo but ran out of time. 

Essentially, the process would be the model performing its inference on the main thread, while the visualizer is displaying the image on a seperate thread. The model would pass its output to the visualizer to output the image. This would clean up the output video, at the cost of predictions being delayed but this may be almost unnoticeable in practice.

This was an engaging project that I thoroughly enjoyed. The main thing that I identified as a future improvement is listed above, others could be a command line interface that allows the user to choose different models, a training script, etc. The library itself does come with these scripts already, but I found there were some adjustments I had to make to get it running.


## Installation:

Clone the repo:
```
git clone https://github.com/husman2012/BasicActionRecognitionProject.git
```
The installation instructions on MMAction Library are quite thorough:
```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
```
## WARNING:
I ran into an issue getting this to work. If the .py file does not run, uninstall mmcv-full and reinstall using ```pip install mmcv-full``` This may take some time to compile.

If all else, fails my environment.yml file that I used to run this is also inside the repo called environment.yml. This will create a conda environment called open-mmcv which should take care of any dependency issues. 

## Running:
This requires a webcam to work. Alternatively, a video path can be input in place of the 0 for ```cv2.VideoCapture```. I've only tested this so far for one person, but I believe it should be robust for multiple people.

python run.py

That should be it. If there are any issues, please let me know.
