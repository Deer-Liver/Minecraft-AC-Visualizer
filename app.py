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
    
# User-configurable parameters
if 'tick_duration' not in st.session_state:
    st.session_state.tick_duration = 0.1  # in seconds
if 'pause_ticks' not in st.session_state:
    st.session_state.pause_ticks = 2  # number of ticks to pause at peaks
if 'voltage_per_tick' not in st.session_state:
    st.session_state.voltage_per_tick = 1.0  # voltage change per tick

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
The wave increases and decreases linearly with configurable timing and voltage rate settings.
You can adjust the tick duration, pause length, and voltage change rate using the settings below.
""")

# Parameters section
st.subheader("Wave Parameter Settings")
params_col1, params_col2, params_col3 = st.columns(3)

# Only allow changing parameters when the generator is stopped
with params_col1:
    st.number_input("Tick Duration (seconds)", 
                   min_value=0.01, 
                   max_value=1.0, 
                   value=st.session_state.tick_duration,
                   step=0.01,
                   key="tick_duration",
                   disabled=st.session_state.running,
                   help="Duration of one tick in seconds")

with params_col2:
    st.number_input("Pause Duration (ticks)", 
                   min_value=1, 
                   max_value=10, 
                   value=st.session_state.pause_ticks,
                   step=1,
                   key="pause_ticks",
                   disabled=st.session_state.running,
                   help="Number of ticks to pause at peaks and troughs")

with params_col3:
    st.number_input("Voltage Change per Tick", 
                   min_value=0.1, 
                   max_value=5.0, 
                   value=st.session_state.voltage_per_tick,
                   step=0.1,
                   key="voltage_per_tick",
                   disabled=st.session_state.running,
                   help="Amount of voltage to change per tick")

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
        ticks_remaining = st.session_state.pause_ticks - st.session_state.pause_counter
        time_remaining = ticks_remaining * st.session_state.tick_duration
        st.metric(label="Pause Remaining", value=f"{time_remaining:.2f}s ({ticks_remaining} ticks)")
    else:
        st.metric(label="Pause Remaining", value="N/A")

# Visualization
st.subheader("Waveform Visualization")

# Create plot
fig = go.Figure()

# Add the waveform data if it exists
if not st.session_state.data.empty:
    # Plot a single continuous line for the wave
    fig.add_trace(go.Scatter(
        x=st.session_state.data['time'],
        y=st.session_state.data['voltage'],
        mode='lines',
        name='Triangle Wave',
        line=dict(color='blue', width=2)
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

# Parameter Information
st.subheader("Wave Parameters Info")
st.markdown(f"""
- **Amplitude Range**: -15V to +15V
- **Time Resolution**: {st.session_state.tick_duration} seconds per tick
- **Pause Duration**: {st.session_state.pause_ticks} ticks at peaks and troughs ({st.session_state.pause_ticks * st.session_state.tick_duration:.2f} seconds)
- **Rise/Fall Rate**: {st.session_state.voltage_per_tick} volts per tick
""")

# Function to update wave data
def update_wave_data():
    current_time = datetime.now()
    if st.session_state.start_time is None:
        st.session_state.start_time = current_time
    
    # Calculate elapsed time based on tick duration
    st.session_state.elapsed_time = (len(st.session_state.data) * st.session_state.tick_duration) if not st.session_state.data.empty else 0
    
    # Update voltage based on current state
    if st.session_state.state == 'increasing':
        st.session_state.voltage += st.session_state.voltage_per_tick
        if st.session_state.voltage >= 15:
            st.session_state.voltage = 15
            st.session_state.state = 'pause_high'
            st.session_state.pause_counter = 0
    
    elif st.session_state.state == 'pause_high':
        st.session_state.pause_counter += 1
        if st.session_state.pause_counter >= st.session_state.pause_ticks:
            st.session_state.state = 'decreasing'
            st.session_state.pause_counter = 0
    
    elif st.session_state.state == 'decreasing':
        st.session_state.voltage -= st.session_state.voltage_per_tick
        if st.session_state.voltage <= -15:
            st.session_state.voltage = -15
            st.session_state.state = 'pause_low'
            st.session_state.pause_counter = 0
    
    elif st.session_state.state == 'pause_low':
        st.session_state.pause_counter += 1
        if st.session_state.pause_counter >= st.session_state.pause_ticks:
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
    time.sleep(st.session_state.tick_duration)  # Sleep for configured tick duration
    st.rerun()  # Rerun the app to update the UI
