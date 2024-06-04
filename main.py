import cv2
import numpy as np

cap = cv2.VideoCapture('input_video.mp4')

# video properties
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Saving the processed video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))


_, prev = cap.read() #first frame
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) #color to grayscale for frame

# empty transformation matrix
transforms = np.zeros((n_frames-1, 3), np.float32)

for i in range(n_frames-1):
    success, curr = cap.read() #reading frame in loops
    if not success:
        break
    
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    
    # tracking feature points with optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    # Filtering only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    # Finding transformation matrix
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    
    # translation and rotation angle
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])
    
    transforms[i] = [dx, dy, da] #storing into empty transformation matrix
    prev_gray = curr_gray

# finding trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

smoothed_trajectory = np.copy(trajectory)
radius = 20
for i in range(radius, n_frames-radius):
    smoothed_trajectory[i] = np.mean(trajectory[i-radius:i+radius], axis=0)

# difference in trajectory to get stabilization transforms
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write stabilized video
for i in range(n_frames-1):
    success, frame = cap.read()
    if not success:
        break
    
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]
    
    m = np.array([[np.cos(da), -np.sin(da), dx], 
                  [np.sin(da),  np.cos(da), dy]])  #creating transformation matrix
    
    # Apply transformation
    frame_stabilized = cv2.warpAffine(frame, m, (width, height))
    out.write(frame_stabilized)


cap.release()
out.release()

print("Video stabilization complete.")
