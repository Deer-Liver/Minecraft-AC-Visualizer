import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import math
import json
import base64
import io
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Wave Generator",
    page_icon="📈",
    layout="wide"
)

# Define themes
THEMES = {
    "Default": {
        "background_color": "#FFFFFF",
        "text_color": "#000000",
        "wave_color": "#1E90FF",
        "grid_color": "#CCCCCC",
        "marker_color": "#000000",
        "ref_line_high": "rgba(255,0,0,0.3)",
        "ref_line_low": "rgba(0,0,255,0.3)"
    },
    "Dark": {
        "background_color": "#121212",
        "text_color": "#FFFFFF",
        "wave_color": "#00BFFF",
        "grid_color": "#333333",
        "marker_color": "#FFFFFF",
        "ref_line_high": "rgba(255,0,0,0.3)",
        "ref_line_low": "rgba(0,0,255,0.3)"
    },
    "Retro": {
        "background_color": "#000033",
        "text_color": "#00FF00",
        "wave_color": "#00FF00",
        "grid_color": "#003300",
        "marker_color": "#FFFF00",
        "ref_line_high": "rgba(255,255,0,0.3)",
        "ref_line_low": "rgba(0,255,0,0.3)"
    },
    "Pastel": {
        "background_color": "#F2F4F8",
        "text_color": "#555555",
        "wave_color": "#FF6B6B",
        "grid_color": "#DEDEDE",
        "marker_color": "#7971EA",
        "ref_line_high": "rgba(255,107,107,0.3)",
        "ref_line_low": "rgba(121,113,234,0.3)"
    }
}

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
if 'amplitude' not in st.session_state:
    st.session_state.amplitude = 15
if 'step_time' not in st.session_state:
    st.session_state.step_time = 0.1
if 'pause_time' not in st.session_state:
    st.session_state.pause_time = 0.2
if 'step_voltage' not in st.session_state:
    st.session_state.step_voltage = 1
if 'wave_type' not in st.session_state:
    st.session_state.wave_type = 'Triangle'
if 'theme' not in st.session_state:
    st.session_state.theme = 'Default'

# Function to reset the generator
def reset_generator():
    st.session_state.running = False
    st.session_state.data = pd.DataFrame({'time': [], 'voltage': [], 'state': []})
    st.session_state.start_time = None
    st.session_state.elapsed_time = 0
    st.session_state.voltage = -st.session_state.amplitude
    st.session_state.direction = 'up'
    st.session_state.state = 'increasing'
    st.session_state.pause_counter = 0
    
# Function to apply theme settings to the UI
def apply_theme(theme_name):
    st.session_state.theme = theme_name
    return THEMES[theme_name]
    
# Function to get a downloadable link for data export
def get_download_link(data, filename, text):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to export the plot as an image
def export_plot_as_image(fig, theme_colors):
    # Create a matplotlib plot for export with theme colors
    plt.figure(figsize=(10, 6), facecolor=theme_colors["background_color"])
    ax = plt.gca()
    ax.set_facecolor(theme_colors["background_color"])
    
    # We'll use matplotlib instead of plotly for export due to better theme handling
    if not st.session_state.data.empty:
        plt.plot(st.session_state.data['time'], st.session_state.data['voltage'], 
                color=theme_colors["wave_color"],
                linewidth=2)
    
    # Set title and labels with theme text color
    plt.title(f"{st.session_state.wave_type} Wave", color=theme_colors["text_color"])
    plt.xlabel("Time (seconds)", color=theme_colors["text_color"])
    plt.ylabel("Voltage (V)", color=theme_colors["text_color"])
    
    # Add reference lines for amplitude
    plt.axhline(y=st.session_state.amplitude, 
               color=theme_colors["ref_line_high"].replace("rgba", "rgb").replace(",0.3)", ")"), 
               linestyle='--', alpha=0.5)
    plt.axhline(y=-st.session_state.amplitude, 
               color=theme_colors["ref_line_low"].replace("rgba", "rgb").replace(",0.3)", ")"), 
               linestyle='--', alpha=0.5)
    
    # Configure grid with theme grid color
    plt.grid(True, color=theme_colors["grid_color"], alpha=0.5)
    
    # Set tick colors to match theme
    plt.tick_params(colors=theme_colors["text_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(theme_colors["text_color"])
    
    # Set y-axis limits
    plt.ylim(-st.session_state.amplitude-1, st.session_state.amplitude+1)
    
    plt.tight_layout()
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=theme_colors["background_color"])
    buf.seek(0)
    
    # Get the bytes
    img_bytes = buf.getvalue()
    plt.close()
    
    return img_bytes

# Function to toggle the generator's running state
def toggle_generator():
    if st.session_state.running:
        st.session_state.running = False
    else:
        st.session_state.running = True
        if st.session_state.start_time is None:
            st.session_state.start_time = datetime.now()

# Apply current theme
current_theme = apply_theme(st.session_state.theme)

# Main title and description
st.title(f"📈 Wave Generator - {st.session_state.wave_type} Wave")
st.markdown(f"""
This application generates a {st.session_state.wave_type.lower()} wave signal that oscillates between -{st.session_state.amplitude}V and +{st.session_state.amplitude}V.
""")

# Create sidebar for parameters and theme
with st.sidebar:
    st.header("⚙️ Wave Settings")
    
    # Wave type selection
    wave_type = st.selectbox(
        "Wave Type",
        options=["Triangle", "Sine", "Square", "Sawtooth"],
        index=["Triangle", "Sine", "Square", "Sawtooth"].index(st.session_state.wave_type)
    )
    if wave_type != st.session_state.wave_type:
        st.session_state.wave_type = wave_type
        reset_generator()
    
    # Amplitude adjustment (ensure it's a numeric type)
    if isinstance(st.session_state.amplitude, list):
        st.session_state.amplitude = 15
        
    amplitude = st.slider(
        "Amplitude (V)",
        min_value=1,
        max_value=30,
        value=int(st.session_state.amplitude),
        step=1
    )
    if amplitude != st.session_state.amplitude:
        st.session_state.amplitude = amplitude
        # Reset if value changed, so the wave starts with the new amplitude
        if not st.session_state.running:
            reset_generator()
    
    # Time settings
    st.subheader("Timing Parameters")
    
    # Ensure step_time is a float
    if isinstance(st.session_state.step_time, list):
        st.session_state.step_time = 0.1
        
    step_time = st.slider(
        "Step Time (s)",
        min_value=0.05,
        max_value=0.5,
        value=float(st.session_state.step_time),
        step=0.05,
        format="%.2f"
    )
    if step_time != st.session_state.step_time:
        st.session_state.step_time = step_time
    
    # Ensure pause_time is a float
    if isinstance(st.session_state.pause_time, list):
        st.session_state.pause_time = 0.2
        
    pause_time = st.slider(
        "Pause Time at Peaks (s)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.pause_time),
        step=0.1,
        format="%.1f"
    )
    if pause_time != st.session_state.pause_time:
        st.session_state.pause_time = pause_time
    
    # Ensure step_voltage is a float, not a list
    if isinstance(st.session_state.step_voltage, list):
        st.session_state.step_voltage = 1.0
        
    step_voltage = st.slider(
        "Voltage Change per Step",
        min_value=0.1,
        max_value=5.0,
        value=float(st.session_state.step_voltage),
        step=0.1,
        format="%.1f"
    )
    if step_voltage != st.session_state.step_voltage:
        st.session_state.step_voltage = step_voltage
    
    # Theme selection
    st.subheader("🎨 Theme Selection")
    theme_selection = st.selectbox(
        "Select Theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme)
    )
    if theme_selection != st.session_state.theme:
        current_theme = apply_theme(theme_selection)
    
    # Export options
    st.subheader("📊 Export Options")
    if not st.session_state.data.empty:
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_button_key = f"csv_button_{int(time.time())}"
            if st.button("Export Data (CSV)", key=csv_button_key, use_container_width=True):
                download_link = get_download_link(
                    st.session_state.data,
                    f"{st.session_state.wave_type.lower()}_wave_data.csv",
                    "📥 Download CSV"
                )
                st.markdown(download_link, unsafe_allow_html=True)
        
        with export_col2:
            plot_button_key = f"plot_button_{int(time.time())}"
            if st.button("Generate Plot Image", key=plot_button_key, use_container_width=True):
                # Get current theme colors
                current_theme_colors = THEMES[st.session_state.theme]
                
                # Create a matplotlib plot for export with theme colors
                plt.figure(figsize=(10, 6), facecolor=current_theme_colors["background_color"])
                ax = plt.gca()
                ax.set_facecolor(current_theme_colors["background_color"])
                
                # Plot the wave
                plt.plot(st.session_state.data['time'], st.session_state.data['voltage'], 
                        color=current_theme_colors["wave_color"],
                        linewidth=2)
                
                # Set title and labels with theme text color
                plt.title(f"{st.session_state.wave_type} Wave", color=current_theme_colors["text_color"])
                plt.xlabel("Time (seconds)", color=current_theme_colors["text_color"])
                plt.ylabel("Voltage (V)", color=current_theme_colors["text_color"])
                
                # Add reference lines for amplitude
                plt.axhline(y=st.session_state.amplitude, 
                           color=current_theme_colors["ref_line_high"].replace("rgba", "rgb").replace(",0.3)", ")"), 
                           linestyle='--', alpha=0.5)
                plt.axhline(y=-st.session_state.amplitude, 
                           color=current_theme_colors["ref_line_low"].replace("rgba", "rgb").replace(",0.3)", ")"), 
                           linestyle='--', alpha=0.5)
                
                # Configure grid with theme grid color
                plt.grid(True, color=current_theme_colors["grid_color"], alpha=0.5)
                
                # Set tick colors to match theme
                plt.tick_params(colors=current_theme_colors["text_color"])
                for spine in ax.spines.values():
                    spine.set_edgecolor(current_theme_colors["text_color"])
                
                # Set y-axis limits
                plt.ylim(-st.session_state.amplitude-1, st.session_state.amplitude+1)
                
                plt.tight_layout()
                
                # Save to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', facecolor=current_theme_colors["background_color"])
                buf.seek(0)
                plt.close()
                
                # Create a download link
                b64 = base64.b64encode(buf.read()).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="{st.session_state.wave_type.lower()}_wave_plot.png">📥 Download PNG</a>'
                st.markdown(href, unsafe_allow_html=True)

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
        pause_ticks = int(st.session_state.pause_time / st.session_state.step_time)
        if pause_ticks > 0:
            pause_remaining = st.session_state.pause_time - (st.session_state.pause_counter * st.session_state.step_time)
            st.metric(label="Pause Remaining", value=f"{pause_remaining:.2f}s")
        else:
            st.metric(label="Pause Remaining", value="0.00s")
    else:
        st.metric(label="Pause Remaining", value="N/A")

# Visualization
st.subheader("Waveform Visualization")

# Create plot
fig = go.Figure()

# Apply theme colors to the plot
theme_colors = THEMES[st.session_state.theme]

# Add the waveform data if it exists
if not st.session_state.data.empty:
    # Plot a single continuous line for the wave
    fig.add_trace(go.Scatter(
        x=st.session_state.data['time'],
        y=st.session_state.data['voltage'],
        mode='lines',
        name=f"{st.session_state.wave_type} Wave",
        line=dict(color=theme_colors["wave_color"], width=2)
    ))

# Configure layout with theme colors
fig.update_layout(
    title=f"{st.session_state.wave_type} Wave",
    xaxis_title="Time (seconds)",
    yaxis_title="Voltage (V)",
    plot_bgcolor=theme_colors["background_color"],
    paper_bgcolor=theme_colors["background_color"],
    font=dict(color=theme_colors["text_color"]),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor=theme_colors["grid_color"],
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor=theme_colors["grid_color"],
        range=[-st.session_state.amplitude-1, st.session_state.amplitude+1]  # Set y-axis limits based on amplitude
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

# Add reference lines at -amplitude and +amplitude
fig.add_shape(
    type="line",
    x0=0, 
    y0=-st.session_state.amplitude, 
    x1=max(st.session_state.data['time'].tolist() + [10]), 
    y1=-st.session_state.amplitude,
    line=dict(color=theme_colors["ref_line_low"], width=1, dash="dash")
)
fig.add_shape(
    type="line",
    x0=0, 
    y0=st.session_state.amplitude, 
    x1=max(st.session_state.data['time'].tolist() + [10]), 
    y1=st.session_state.amplitude,
    line=dict(color=theme_colors["ref_line_high"], width=1, dash="dash")
)

# Add the current point as a dot if the generator is running
if st.session_state.running and not st.session_state.data.empty:
    last_time = st.session_state.data['time'].iloc[-1]
    last_voltage = st.session_state.data['voltage'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_time],
        y=[last_voltage],
        mode='markers',
        marker=dict(color=theme_colors["marker_color"], size=10),
        showlegend=False
    ))

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Information section
st.subheader("Wave Parameters")
parameter_info = f"""
- **Wave Type**: {st.session_state.wave_type}
- **Amplitude Range**: -{st.session_state.amplitude}V to +{st.session_state.amplitude}V
- **Time Resolution**: {st.session_state.step_time} seconds per tick
- **Pause Duration**: {st.session_state.pause_time} seconds at peaks and troughs
- **Rise/Fall Rate**: {st.session_state.step_voltage} volt per tick
"""
st.markdown(parameter_info)

# Calculate sine wave value
def get_sine_wave_value(time, amplitude):
    return amplitude * math.sin(2 * math.pi * time)

# Calculate square wave value
def get_square_wave_value(time, amplitude):
    sine_value = math.sin(2 * math.pi * time)
    return amplitude if sine_value >= 0 else -amplitude

# Calculate sawtooth wave value 
def get_sawtooth_wave_value(time, amplitude):
    # Period is 1 second for simplicity
    return (2 * amplitude * (time % 1)) - amplitude

# Function to update wave data
def update_wave_data():
    current_time = datetime.now()
    if st.session_state.start_time is None:
        st.session_state.start_time = current_time
    
    # Calculate elapsed time
    st.session_state.elapsed_time = (len(st.session_state.data) * st.session_state.step_time) if not st.session_state.data.empty else 0
    
    # Different behavior based on wave type
    if st.session_state.wave_type == "Triangle":
        # Update voltage based on current state for triangle wave
        if st.session_state.state == 'increasing':
            st.session_state.voltage += st.session_state.step_voltage
            if st.session_state.voltage >= st.session_state.amplitude:
                st.session_state.voltage = st.session_state.amplitude
                # Only pause if pause time is > 0
                if st.session_state.pause_time > 0:
                    st.session_state.state = 'pause_high'
                    st.session_state.pause_counter = 0
                else:
                    st.session_state.state = 'decreasing'
        
        elif st.session_state.state == 'pause_high':
            st.session_state.pause_counter += 1
            pause_ticks = int(st.session_state.pause_time / st.session_state.step_time)
            if st.session_state.pause_counter >= pause_ticks:
                st.session_state.state = 'decreasing'
                st.session_state.pause_counter = 0
        
        elif st.session_state.state == 'decreasing':
            st.session_state.voltage -= st.session_state.step_voltage
            if st.session_state.voltage <= -st.session_state.amplitude:
                st.session_state.voltage = -st.session_state.amplitude
                # Only pause if pause time is > 0
                if st.session_state.pause_time > 0:
                    st.session_state.state = 'pause_low'
                    st.session_state.pause_counter = 0
                else:
                    st.session_state.state = 'increasing'
        
        elif st.session_state.state == 'pause_low':
            st.session_state.pause_counter += 1
            pause_ticks = int(st.session_state.pause_time / st.session_state.step_time)
            if st.session_state.pause_counter >= pause_ticks:
                st.session_state.state = 'increasing'
                st.session_state.pause_counter = 0
    
    elif st.session_state.wave_type == "Sine":
        # Calculate sine wave value based on elapsed time
        time_period = 4 * st.session_state.step_time / st.session_state.step_voltage
        st.session_state.voltage = get_sine_wave_value(st.session_state.elapsed_time / time_period, st.session_state.amplitude)
        # Update state for display purposes
        if st.session_state.voltage > 0 and st.session_state.voltage < st.session_state.amplitude:
            st.session_state.state = 'increasing'
        elif st.session_state.voltage < 0 and st.session_state.voltage > -st.session_state.amplitude:
            st.session_state.state = 'decreasing'
        elif abs(st.session_state.voltage) >= st.session_state.amplitude * 0.98:
            st.session_state.state = 'pause_high' if st.session_state.voltage > 0 else 'pause_low'
    
    elif st.session_state.wave_type == "Square":
        # Calculate square wave value
        time_period = 4 * st.session_state.step_time / st.session_state.step_voltage
        st.session_state.voltage = get_square_wave_value(st.session_state.elapsed_time / time_period, st.session_state.amplitude)
        # Update state based on the value
        if st.session_state.voltage > 0:
            st.session_state.state = 'pause_high'
        else:
            st.session_state.state = 'pause_low'
    
    elif st.session_state.wave_type == "Sawtooth":
        # Calculate sawtooth wave value
        time_period = 2 * st.session_state.amplitude / st.session_state.step_voltage * st.session_state.step_time
        st.session_state.voltage = get_sawtooth_wave_value(st.session_state.elapsed_time / time_period, st.session_state.amplitude)
        # Update state for display
        if st.session_state.voltage < st.session_state.amplitude * 0.9:
            st.session_state.state = 'increasing'
        else:
            st.session_state.state = 'pause_high'
    
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
    time.sleep(st.session_state.step_time)  # Sleep based on configured step time
    st.rerun()  # Rerun the app to update the UI
