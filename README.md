# Video-Stabilization-Python

### Libraries:

```python
pip install numpy
pip install opencv-python
```

Please open this [Drive Link](https://drive.google.com/drive/folders/1nbgzpcC_qOFp0WAKy8hhFYAcVKQ7FDzn?usp=drive_link) for viewing the stable videos, I got the videos which are recorded with shaky camera and make it stable using python opencv.

What I have done:
1. Added the input video file
2. Using Opencv properties/functions I fetched out the video properties
3. Wrote the code to save the processed video file with mp4v video codec.
4. Considered first frame for initial reference. And also converting the frame into grayscale.
5. Created transformation matrix using numpy it will be empty.
6. Now looping through each frame of video reading the frame, converting to grayscale, tracking the feature points using lucas-kanade optical flow, filter out invalid points, affine transformation matrix and then extract translation and rotation from that matrix and store it in empty transformation matrix then applies to moving filters to smooth out the trajectory.
7. Calculate cumulative sum of transformation matrix, then find out the difference between smoothed trajectory and original trajectory.
8. Reset the video stream and using loop which will be running through each frame where we will be applying the smoothed transformation which helps to stabilized frame then write the output video.
9. Getting the stable video as output by releasing the resources.
