# Initialize MediaPipe Pose model
body_part_index = 32
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(VIDEO_PATH)

# Create an empty list to store the trace of the right elbow
trace = []

# Create empty lists to store the x, y, z coordinates of the right elbow
x_vals = []
y_vals = []
z_vals = []

# Create a Matplotlib figure and subplot for the real-time updating plot
# fig, ax = plt.subplots()
# plt.title('Time Lapse of the X Coordinate')
# plt.xlabel('Frames')
# plt.ylabel('Coordinate Value')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.ion()
# plt.show()
frame_num = 0

while True:
    # Read a frame from the video capture
    success, image = cap.read()
    if not success:
        break
    # Convert the frame to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose model
    results = pose.process(image)

    # Check if any body parts are detected
    
    if results.pose_landmarks:
        # Get the x,y,z coordinates of the right elbow
        x, y, z = results.pose_landmarks.landmark[body_part_index].x, results.pose_landmarks.landmark[body_part_index].y, results.pose_landmarks.landmark[body_part_index].z
        
        # Append the x, y, z values to the corresponding lists
        #x_vals.append(x)
        y_vals.append(y)
        #z_vals.append(z)
        
        # # Add the (x, y) coordinates to the trace list
        trace.append((int(x * image.shape[1]), int(y * image.shape[0])))

        # Draw the trace on the image
        for i in range(len(trace)-1):
            cv2.line(image, trace[i], trace[i+1], (255, 0, 0), thickness=2)

        plt.title('Time Lapse of the Y Coordinate')
        plt.xlabel('Frames')
        plt.ylabel('Coordinate Value')
        plt.xlim(0,len(pose_coords))
        plt.ylim(0,1)
        plt.plot(y_vals);
        # Clear the plot and update with the new x, y, z coordinate values
        #ax.clear()
        # ax.plot(range(0, frame_num + 1), x_vals, 'r.', label='x')
        # ax.plot(range(0, frame_num + 1), y_vals, 'g.', label='y')
        # ax.plot(range(0, frame_num + 1), z_vals, 'b.', label='z')
        # ax.legend(loc='upper left')
        # plt.draw()
        plt.pause(0.00000000001)
        clear_output(wait=True)
        frame_num += 1
    
    # Convert the image back to BGR format for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow('Pose Tracking', image)

    # Wait for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the video capture, close all windows, and clear the plot
cap.release()
cv2.destroyAllWindows()
plt.close()