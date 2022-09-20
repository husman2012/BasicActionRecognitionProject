Utilizes MMAction Library from Open-MM
Link: https://github.com/open-mmlab/mmaction2

The main reason that I chose to use this library in particular is that it has a plethora of models available for testing and is also able to accept more. With the short time available for this project, I thought it was best to achieve the Minimum Viable Product, rather than completely build from scratch. This library has several models, including transformer models, which are pre-trained for action recognition. The process we use is based upon the SpatioTemporal Action Detection Webcam Demo. 

This utilizes a 3D CNN accepting 8 frames of input at a given time. SpatioTemporal Models have been shown to be exceptional within this problem set. However, this comes at somewhat of a cost in real-time applications. While this project still runs at real-time, it is very choppy as it must wait for model output before drawing. Originally, I had planned to multithread this, similar to the demo but ran out of time. 

Essentially, the process would be the model performing its inference on the main thread, while the visualizer is displaying the image on a seperate thread. The model would pass its output to the visualizer to output the image. This would clean up the output video, at the cost of predictions being delayed but this may be almost unnoticeable in practice.

This was an engaging project that I thoroughly enjoyed. The main thing that I identified as a future improvement are listed above, others could be a command line interface that allows the user to choose different models, a training script, etc. The library itself does come with these scripts already, but I found there were some adjustments I had to make to get it running.


Installation:
