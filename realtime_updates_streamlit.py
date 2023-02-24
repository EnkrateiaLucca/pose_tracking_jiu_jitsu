import streamlit as st
import time
import numpy as np
import pandas as pd

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
random_probs = np.random.random((1,3)).ravel()
chart = st.empty()
for i in range(1, 100):
    # update the bar chart
    random_probs = np.random.random(3)
    # turn the random propbablitiles into a pandas dataframe
    chart_data = pd.DataFrame(dict(labels=["x", "y", "z"], values=random_probs))    
    chart.line_chart(chart_data,x="labels",y="values")
    progress_bar.progress(i)
    time.sleep(0.05)

progress_bar.empty()
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")

# TO add the probabilities for the emotions to the bar charts
# I could see the mid level api of the paz package where I cold
# access those predictions while updating the chart