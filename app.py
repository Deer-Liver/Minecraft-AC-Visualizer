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
    
# View state preservation
if 'view_state' not in st.session_state:
    st.session_state.view_state = {
        'time_window': (0, 10.0),
        'voltage_range': (-16.0, 16.0),
        'autoscale_options': 'Fixed Range'
    }

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
    # Make the plot interactive with drag mode set to pan
    dragmode='pan'
)

# Add reference lines at -15V and +15V
x1_value = 10.0
if not st.session_state.data.empty:
    x1_value = max(max(st.session_state.data['time'].tolist()), 10.0)

fig.add_shape(
    type="line",
    x0=0, y0=-15, x1=x1_value, y1=-15,
    line=dict(color="rgba(0,0,255,0.3)", width=1, dash="dash")
)
fig.add_shape(
    type="line",
    x0=0, y0=15, x1=x1_value, y1=15,
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

# Add custom view control options
st.subheader("Visualization Controls")
view_col1, view_col2, view_col3 = st.columns(3)

# Helper function to update view state
def update_view_state(key, value):
    st.session_state.view_state[key] = value

# Initialize time_window and max_time variables to handle empty data case
time_window = st.session_state.view_state['time_window']
min_time = 0.0  # Use float instead of int
max_time = 10.0

# Time window control
with view_col1:
    if not st.session_state.data.empty:
        max_time = max(max(st.session_state.data['time'].tolist()), 10.0)
    
    # Use the saved view state for initial value, ensuring all values are float
    time_window = st.slider(
        "Time Window (seconds)",
        min_value=float(min_time),
        max_value=float(max_time),
        value=(float(time_window[0]), float(time_window[1])) if isinstance(time_window, (list, tuple)) else (0.0, float(max_time)),
        help="Adjust the visible time range",
        on_change=update_view_state,
        args=('time_window',),
        key="time_window_slider"
    )
    st.session_state.view_state['time_window'] = time_window
    # Update x-axis range based on slider
    fig.update_layout(xaxis=dict(range=time_window))

# Voltage range control
with view_col2:
    # Use the saved view state for initial value, ensuring all values are float
    voltage_range = st.slider(
        "Voltage Range",
        min_value=-20.0,
        max_value=20.0,
        value=(float(st.session_state.view_state['voltage_range'][0]), float(st.session_state.view_state['voltage_range'][1])) 
              if isinstance(st.session_state.view_state['voltage_range'], (list, tuple)) 
              else (-16.0, 16.0),
        help="Adjust the visible voltage range",
        on_change=update_view_state,
        args=('voltage_range',),
        key="voltage_range_slider"
    )
    st.session_state.view_state['voltage_range'] = voltage_range
    # Update y-axis range based on slider
    fig.update_layout(yaxis=dict(range=voltage_range))

# Auto-scale options
with view_col3:
    # Use the saved view state for initial value
    autoscale_options = st.radio(
        "Auto-scale Options",
        ["Fixed Range", "Auto-scale X", "Auto-scale Y", "Auto-scale Both"],
        index=["Fixed Range", "Auto-scale X", "Auto-scale Y", "Auto-scale Both"].index(st.session_state.view_state['autoscale_options']),
        help="Choose how the chart scales",
        on_change=update_view_state,
        args=('autoscale_options',),
        key="autoscale_options_radio"
    )
    st.session_state.view_state['autoscale_options'] = autoscale_options
    
    # Apply auto-scaling based on selection
    if autoscale_options == "Auto-scale X":
        fig.update_layout(xaxis=dict(autorange=True), yaxis=dict(range=voltage_range))
    elif autoscale_options == "Auto-scale Y":
        fig.update_layout(xaxis=dict(range=time_window), yaxis=dict(autorange=True))
    elif autoscale_options == "Auto-scale Both":
        fig.update_layout(xaxis=dict(autorange=True), yaxis=dict(autorange=True))
    # "Fixed Range" uses the ranges set by the sliders

# Display the plot with improved configuration
st.plotly_chart(fig, use_container_width=True, config={
    'scrollZoom': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape']
})

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

# Use a container for the updatable content to prevent scroll reset
if 'update_container' not in st.session_state:
    st.session_state.update_container = st.empty()

# Make the app update only when necessary
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

# Main loop to update the wave if running
if st.session_state.running:
    # Only update at the tick rate to prevent constant refreshing
    current_time = time.time()
    if current_time - st.session_state.last_update_time >= st.session_state.tick_duration:
        update_wave_data()
        st.session_state.last_update_time = current_time
        st.rerun()  # Rerun the app to update the UI but maintain scroll position
