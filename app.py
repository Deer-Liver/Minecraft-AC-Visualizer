import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Triangle Wave Generator",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({'time': [], 'voltage': [], 'state': []})
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0
if 'voltage' not in st.session_state:
    st.session_state.voltage = -15
if 'direction' not in st.session_state:
    st.session_state.direction = 'up'
if 'state' not in st.session_state:
    st.session_state.state = 'increasing'
if 'pause_counter' not in st.session_state:
    st.session_state.pause_counter = 0

# Function to reset the generator
def reset_generator():
    st.session_state.running = False
    st.session_state.data = pd.DataFrame({'time': [], 'voltage': [], 'state': []})
    st.session_state.start_time = None
    st.session_state.elapsed_time = 0
    st.session_state.voltage = -15
    st.session_state.direction = 'up'
    st.session_state.state = 'increasing'
    st.session_state.pause_counter = 0

# Function to toggle the generator's running state
def toggle_generator():
    if st.session_state.running:
        st.session_state.running = False
    else:
        st.session_state.running = True
        if st.session_state.start_time is None:
            st.session_state.start_time = datetime.now()

# Main title and description
st.title("📈 Triangle Wave Generator")
st.markdown("""
This application generates a triangle wave signal that oscillates between -15V and +15V.
The wave increases and decreases linearly at a rate of 1 unit per 0.1 second, with pauses at peaks and troughs.
""")

# Control panel
st.subheader("Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start/Stop", use_container_width=True):
        toggle_generator()

with col2:
    if st.button("Reset", use_container_width=True):
        reset_generator()

with col3:
    st.markdown("Status: **{}**".format("Running" if st.session_state.running else "Stopped"))

# Current status metrics
st.subheader("Current Status")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(label="Voltage", value=f"{st.session_state.voltage:.1f}V")

with metric_col2:
    elapsed_time_display = f"{st.session_state.elapsed_time:.1f}s"
    st.metric(label="Elapsed Time", value=elapsed_time_display)

with metric_col3:
    states = {
        'increasing': 'Rising ⬆️',
        'decreasing': 'Falling ⬇️',
        'pause_high': 'Pause (High) ⏸️',
        'pause_low': 'Pause (Low) ⏸️'
    }
    st.metric(label="State", value=states.get(st.session_state.state, "Unknown"))

with metric_col4:
    if st.session_state.state.startswith('pause'):
        pause_remaining = 0.2 - (st.session_state.pause_counter * 0.1)
        st.metric(label="Pause Remaining", value=f"{pause_remaining:.1f}s")
    else:
        st.metric(label="Pause Remaining", value="N/A")

# Visualization
st.subheader("Waveform Visualization")

# Create plot
fig = go.Figure()

# Add the waveform data if it exists
if not st.session_state.data.empty:
    # Define colors for different states
    color_map = {
        'increasing': 'green',
        'decreasing': 'red',
        'pause_high': 'orange',
        'pause_low': 'blue'
    }
    
    # Plot each state with its own color
    for state, color in color_map.items():
        state_data = st.session_state.data[st.session_state.data['state'] == state]
        if not state_data.empty:
            fig.add_trace(go.Scatter(
                x=state_data['time'],
                y=state_data['voltage'],
                mode='lines',
                name=state,
                line=dict(color=color, width=2)
            ))

# Configure layout
fig.update_layout(
    title="Triangle Wave",
    xaxis_title="Time (seconds)",
    yaxis_title="Voltage (V)",
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[-16, 16]  # Set y-axis limits just beyond our voltage range
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    height=500,
)

# Add reference lines at -15V and +15V
fig.add_shape(
    type="line",
    x0=0, y0=-15, x1=max(st.session_state.data['time'].tolist() + [10]), y1=-15,
    line=dict(color="rgba(0,0,255,0.3)", width=1, dash="dash")
)
fig.add_shape(
    type="line",
    x0=0, y0=15, x1=max(st.session_state.data['time'].tolist() + [10]), y1=15,
    line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dash")
)

# Add the current point as a dot if the generator is running
if st.session_state.running and not st.session_state.data.empty:
    last_time = st.session_state.data['time'].iloc[-1]
    last_voltage = st.session_state.data['voltage'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_time],
        y=[last_voltage],
        mode='markers',
        marker=dict(color='black', size=10),
        showlegend=False
    ))

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Information section
st.subheader("Wave Parameters")
st.markdown("""
- **Amplitude Range**: -15V to +15V
- **Time Resolution**: 0.1 seconds per tick
- **Pause Duration**: 0.2 seconds (2 ticks) at peaks and troughs
- **Rise/Fall Rate**: 1 volt per tick
""")

# Function to update wave data
def update_wave_data():
    current_time = datetime.now()
    if st.session_state.start_time is None:
        st.session_state.start_time = current_time
    
    # Calculate elapsed time
    st.session_state.elapsed_time = (len(st.session_state.data) * 0.1) if not st.session_state.data.empty else 0
    
    # Update voltage based on current state
    if st.session_state.state == 'increasing':
        st.session_state.voltage += 1
        if st.session_state.voltage >= 15:
            st.session_state.voltage = 15
            st.session_state.state = 'pause_high'
            st.session_state.pause_counter = 0
    
    elif st.session_state.state == 'pause_high':
        st.session_state.pause_counter += 1
        if st.session_state.pause_counter >= 2:  # 2 ticks of 0.1s = 0.2s pause
            st.session_state.state = 'decreasing'
            st.session_state.pause_counter = 0
    
    elif st.session_state.state == 'decreasing':
        st.session_state.voltage -= 1
        if st.session_state.voltage <= -15:
            st.session_state.voltage = -15
            st.session_state.state = 'pause_low'
            st.session_state.pause_counter = 0
    
    elif st.session_state.state == 'pause_low':
        st.session_state.pause_counter += 1
        if st.session_state.pause_counter >= 2:  # 2 ticks of 0.1s = 0.2s pause
            st.session_state.state = 'increasing'
            st.session_state.pause_counter = 0
    
    # Add new data point
    new_data = pd.DataFrame({
        'time': [st.session_state.elapsed_time],
        'voltage': [st.session_state.voltage],
        'state': [st.session_state.state]
    })
    
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

# Main loop to update the wave if running
if st.session_state.running:
    update_wave_data()
    time.sleep(0.1)  # Sleep for 0.1 seconds to simulate the tick
    st.rerun()  # Rerun the app to update the UI
