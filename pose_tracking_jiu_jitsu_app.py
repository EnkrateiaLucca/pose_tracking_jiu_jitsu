from base64 import b64encode
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List # I don't think I need this!
import streamlit as st

# Custom imports
from pose_tracking_utils import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Set up video file uploader
st.title("Pose Tracking Jiu Jitsu App")
video_file = st.file_uploader("Upload a video", type=["mp4"])

if video_file:
    st.video(video_file)

if st.sidebar.button("Create Pose Tracking Video"):
    st.write("Creating pose tracking video....")
    st.write(video_file)
    video_file_path = os.path.join("./videos/",video_file.name)
    # output_path = create_pose_tracking_video(video_file_path)
    # st.write("Pose Tracking Video Created!")
    # st.write(output_path)
    # st.write(os.path.exists(output_path))
    # st.video(output_path)
    
    
    
    


