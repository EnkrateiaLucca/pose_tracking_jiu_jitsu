import os
import imageio
from natsort import natsorted
import pandas as pd
import pathlib
import plotly.express as px
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
# MediaPipe imports
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

VIDEO_PATH = "./training_session_1.mp4"

# Custom variables to set
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


def create_pose_tracking_video(video_path):
    # For webcam input:
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = pathlib.Path(video_path).stem + "_pose.mp4" 
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            # To improve performance, optinally mark the iamge as 
            # not writeable to pass by reference.
            image.flags.writeable = False
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            # Draw the annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Flip the image horizontally for a self-view display.
            out.write(cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
    cap.release()
    out.release()
    print("Pose video created!")
    
    return output_path


def plot_joint_trajectory():
    pass


def plot_landmarks(
    landmark_list,
    connections=None,
):
    if not landmark_list:
        return
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        out_cn = []
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig


def save_compressed_video(save_path):
    """
    Compresses a .mp4 video file using ffmpeg.
    """
    # Compressed video path
    compressed_path = f"{save_path[:-4]}_compressed.mp4"

    os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")
    
    return compressed_path


def create_animation_from_png(folder,anim_output_path='landmarks_plot_animation.mp4'):    
    # Directory where the PNG files are stored
    directory = '.'

    # List all the PNG files in the directory
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    # Sort the files alphabetically (or numerically if the filenames contain numbers)
    png_files = natsorted(png_files)

    # Create a writer object to write the video
    writer = imageio.get_writer(anim_output_path, fps=30)

    # Iterate over the PNG files and add them to the video
    for png_file in png_files:
        # Read the PNG file as a numpy array
        image = imageio.imread(os.path.join(directory, png_file))
        # Add the image to the video
        writer.append_data(image)
        os.remove(png_file)

    # Close the writer object
    writer.close()


def create_landmarks_plot3D_animation(video_path, output_path):      
    i=0
    # For video input
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        mp_drawing.plot_landmarks_and_save(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS, plot_index=i)
        if cv2.waitKey(1) & 0xFF == 27:
          break
        
        i+=1
        

    cap.release()
    create_animation_from_png(folder=".", anim_output_path=output_path)


def create_video_clip(video_path: str, output_path: str, start_time: str, 
                      length: int):
    """
    Creates a video clip from a video file using ffmpeg.
    """
    
    os.system(f"ffmpeg -ss {start_time} -i input.mp4 -t {length} -c:v libx264 -c:a copy {output_path}")
    print(f"clip created at: {output_path}")
    
    return output_path


def get_pose_coords(video_path: str):
    """
    Gets the pose coordinates for a given video and returns
    a list where each element is a pose_landmarks object from
    mediapipe (or something like that...).
    """
    pose_coords = []
    # For webcam input:
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            # To improve performance, optinally mark the iamge as 
            # not writeable to pass by reference.
            image.flags.writeable = False
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            pose_coords.append(results.pose_landmarks)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
    cap.release()
    print("Pose video created!")
    
    return pose_coords


def create_joint_trace(video_path,body_part_index=32, color_rgb=(255,0,0)):
    """
    This function creates a trace of the body part being tracked.
    body_part_index: The index of the body part being tracked.
    video_path: The path to the video being analysed.
    """
    # Initialize MediaPipe Pose modelpose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize OpenCV VideoCapture object to capture video from the camera
    cap = cv2.VideoCapture(video_path)

    # Create an empty list to store the trace of the body part being tracked
    trace = []

    with mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convert the frame to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose model
            results = pose.process(image)

            # Check if any body parts are detected
            if results.pose_landmarks:
                # Get the x,y coordinates of the body part being tracked (in this case, the right elbow)
                x, y = int(results.pose_landmarks.landmark[body_part_index].x * image.shape[1]), int(results.pose_landmarks.landmark[body_part_index].y * image.shape[0])
                
                # Add the coordinates to the trace list
                trace.append((x, y))

                # Draw the trace on the image
                for i in range(len(trace)-1):
                    cv2.line(image, trace[i], trace[i+1], color_rgb, thickness=2)
            
            # Convert the image back to BGR format for display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Display the image
            cv2.imshow('Pose Tracking', image)

            # Wait for user input to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


def create_joint_trace_video(video_path,body_part_index=32, color_rgb=(255,0,0)):
    """
    This function creates a trace of the body part being tracked.
    body_part_index: The index of the body part being tracked.
    video_path: The path to the video being analysed.
    """
    # Initialize MediaPipe Pose modelpose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize OpenCV VideoCapture object to capture video from the camera
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = pathlib.Path(video_path).stem + "_trace.mp4" 
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    # Create an empty list to store the trace of the body part being tracked
    trace = []

    with mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convert the frame to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose model
            results = pose.process(image)

            # Check if any body parts are detected
            if results.pose_landmarks:
                # Get the x,y coordinates of the body part being tracked (in this case, the right elbow)
                x, y = int(results.pose_landmarks.landmark[body_part_index].x * image.shape[1]), int(results.pose_landmarks.landmark[body_part_index].y * image.shape[0])
                
                # Add the coordinates to the trace list
                trace.append((x, y))

                # Draw the trace on the image
                for i in range(len(trace)-1):
                    cv2.line(image, trace[i], trace[i+1], color_rgb, thickness=2)
            
            # Convert the image back to BGR format for display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Display the image
            out.write(image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
    cap.release()
    out.release()
    print("Joint Trace video created!")


def create_joint_trace_graph(video_path, output_path, body_part_index,xmin=300,xmax=1000,
                             ymin=200,ymax=800):
    """
    Creates a graph with the tracing of a particular body part,
    while executing a certain movement.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create an empty list to store the trace of the body part being tracked
    trace = []
    i = 0
    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convert the frame to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose model
            results = pose.process(image)

            # Check if any body parts are detected
            if results.pose_landmarks:
                # Get the x,y coordinates of the body part being tracked (in this case, the right elbow)
                x, y = int(results.pose_landmarks.landmark[body_part_index].x * image.shape[1]), int(results.pose_landmarks.landmark[body_part_index].y * image.shape[0])

                # Add the coordinates to the trace list
                trace.append((x, y))

                # Plot the trace on the graph
                fig, ax = plt.subplots()
                #ax.imshow(image)
                # ax.set_xlim(xmin,xmax)
                # ax.set_ylim(ymin,ymax)
                ax.invert_yaxis()
                ax.plot(np.array(trace)[:, 0], np.array(trace)[:, 1], color='r')
                plt.savefig(f'joint_trace{i}.png')
                plt.close()
                i+=1
                #plt.pause(0.00000000001)
                #clear_output(wait=True)
                # Display the graph
                #plt.show()
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        print("Joint Trace graph created!")
        print("Creating animation")
        create_animation_from_png(".", output_path)


def get_joint_trace_data(video_path, body_part_index,xmin=300,xmax=1000,
                             ymin=200,ymax=800):
    """
    Creates a graph with the tracing of a particular body part,
    while executing a certain movement.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create an empty list to store the trace of the body part being tracked
    trace = []
    i = 0
    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convert the frame to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose model
            results = pose.process(image)

            # Check if any body parts are detected
            if results.pose_landmarks:
                # Get the x,y coordinates of the body part being tracked (in this case, the right elbow)
                x, y = int(results.pose_landmarks.landmark[body_part_index].x * image.shape[1]), int(results.pose_landmarks.landmark[body_part_index].y * image.shape[0])

                # Add the coordinates to the trace list
                trace.append((x, y))

                # Plot the trace on the graph
                fig, ax = plt.subplots()
                #ax.imshow(image)
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                ax.invert_yaxis()
                ax.plot(np.array(trace)[:, 0], np.array(trace)[:, 1], color='r')
                # plt.savefig(f'joint_trace{i}.png')
                # plt.close()
                i+=1
                plt.pause(0.00000000001)
                clear_output(wait=True)
                # Display the graph
                #plt.show()
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        
        return trace
    
        

def find_individual_traces(trace,window_size=60, color_plot="r"):
    """
    Function that takes in a liste of tuples containing x,y coordinates
    and plots them as different clips with varying sizes to allow the user to find
    the point where a full repetition has been completed
    """
    
    clip_size = 0
    for i in range(len(trace)//window_size):
        plt.plot(np.array(trace[clip_size:clip_size+window_size])[:, 0], np.array(trace[clip_size:clip_size+window_size])[:, 1], color=color_plot)
        plt.gca().invert_yaxis()
        plt.title(f"Trace, clip size = {clip_size}")
        plt.show()
        clip_size+=window_size


def get_individual_traces(trace, clip_size):
    num_clips = len(trace)//clip_size
    trace_clips = []
    i = 0
    for clip in range(num_clips):
        trace_clips.append(trace[i:i+clip_size])
        i+=clip_size
    
    return trace_clips


def get_mean_trace(traces):
    return np.mean(np.array(traces),axis=0)