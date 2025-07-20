import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_clm_data(filename):
    """Parse the CLM Logs CSV file"""
    # Read the raw CSV data
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Find the data rows (skip empty rows and headers)
    data_rows = []
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(',Month') and not line.startswith(',,,,'):
            parts = line.strip().split(',')
            if len(parts) > 10 and parts[1] and parts[1] != 'Month':
                data_rows.append(parts)
    
    # Extract PIC MPD and MUX MPD data
    pic_data = []
    mux_data = []
    timestamps = []
    
    for row in data_rows[:13]:  # First 13 rows contain the main data
        if len(row) > 33:
            timestamp = row[2]  # Extract timestamp
            timestamps.append(timestamp)
            
            # PIC MPD data (columns 3-18)
            pic_values = [int(x) if x.isdigit() else 0 for x in row[3:19]]
            pic_data.append(pic_values)
            
            # MUX MPD data (columns 19-34)
            mux_values = [int(x) if x.isdigit() else 0 for x in row[19:35]]
            mux_data.append(mux_values)
    
    return timestamps, pic_data, mux_data

def create_dashboard():
    """Create a comprehensive dashboard for CLM Logs data"""
    # Parse the data
    timestamps, pic_data, mux_data = parse_clm_data('CLM Logs - Bench 1.csv')
    
    # Create figure with subplots - increased figure size and spacing
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('CLM Logs - Bench 1 Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Convert data to numpy arrays for easier manipulation
    pic_array = np.array(pic_data)
    mux_array = np.array(mux_data)
    
    # Plot 1: PIC MPD Heatmap
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(pic_array, cmap='viridis', aspect='auto')
    ax1.set_title('PIC MPD (Bench 1) - Heatmap', fontsize=12, pad=20)
    ax1.set_xlabel('Channel Index (0-15)')
    ax1.set_ylabel('Time Sample')
    ax1.set_yticks(range(len(timestamps)))
    ax1.set_yticklabels([t[:8] for t in timestamps], rotation=45, ha='right')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: MUX MPD Heatmap
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(mux_array, cmap='viridis', aspect='auto')
    ax2.set_title('MUX MPD (Bench 1) - Heatmap', fontsize=12, pad=20)
    ax2.set_xlabel('Channel Index (0-15)')
    ax2.set_ylabel('Time Sample')
    ax2.set_yticks(range(len(timestamps)))
    ax2.set_yticklabels([t[:8] for t in timestamps], rotation=45, ha='right')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Average values over time
    ax3 = plt.subplot(3, 3, 3)
    pic_avg = np.mean(pic_array, axis=1)
    mux_avg = np.mean(mux_array, axis=1)
    x_pos = range(len(timestamps))
    ax3.plot(x_pos, pic_avg, 'o-', label='PIC MPD Avg', linewidth=2, markersize=6)
    ax3.plot(x_pos, mux_avg, 's-', label='MUX MPD Avg', linewidth=2, markersize=6)
    ax3.set_title('Average Values Over Time', fontsize=12, pad=20)
    ax3.set_xlabel('Time Sample')
    ax3.set_ylabel('Average Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos[::2])
    ax3.set_xticklabels([timestamps[i][:8] for i in x_pos[::2]], rotation=45)
    
    # Plot 4: Channel comparison (latest values)
    ax4 = plt.subplot(3, 3, 4)
    channels = list(range(16))
    latest_pic = pic_array[-1]
    latest_mux = mux_array[-1]
    
    x = np.arange(len(channels))
    width = 0.35
    ax4.bar(x - width/2, latest_pic, width, label='PIC MPD', alpha=0.8)
    ax4.bar(x + width/2, latest_mux, width, label='MUX MPD', alpha=0.8)
    ax4.set_title(f'Latest Values by Channel ({timestamps[-1][:8]})', fontsize=12, pad=20)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(channels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Value distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(pic_array.flatten(), bins=30, alpha=0.7, label='PIC MPD', density=True)
    ax5.hist(mux_array.flatten(), bins=30, alpha=0.7, label='MUX MPD', density=True)
    ax5.set_title('Value Distribution', fontsize=12, pad=20)
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Trend analysis for specific channels
    ax6 = plt.subplot(3, 3, 6)
    # Plot trends for channels 0, 5, 10, 15
    channels_to_plot = [0, 5, 10, 15]
    for ch in channels_to_plot:
        ax6.plot(x_pos, pic_array[:, ch], 'o-', label=f'PIC Ch{ch}', alpha=0.7)
        ax6.plot(x_pos, mux_array[:, ch], 's--', label=f'MUX Ch{ch}', alpha=0.7)
    
    ax6.set_title('Channel Trends (Selected Channels)', fontsize=12, pad=20)
    ax6.set_xlabel('Time Sample')
    ax6.set_ylabel('Value')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    ax6.set_xticks(x_pos[::2])
    ax6.set_xticklabels([timestamps[i][:8] for i in x_pos[::2]], rotation=45)
    
    # Add two more plots to fill the 3x3 grid
    
    # Plot 7: Max values over time
    ax7 = plt.subplot(3, 3, 7)
    pic_max = np.max(pic_array, axis=1)
    mux_max = np.max(mux_array, axis=1)
    ax7.plot(x_pos, pic_max, 'o-', label='PIC MPD Max', linewidth=2, markersize=6)
    ax7.plot(x_pos, mux_max, 's-', label='MUX MPD Max', linewidth=2, markersize=6)
    ax7.set_title('Maximum Values Over Time', fontsize=12, pad=20)
    ax7.set_xlabel('Time Sample')
    ax7.set_ylabel('Maximum Value')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks(x_pos[::2])
    ax7.set_xticklabels([timestamps[i][:8] for i in x_pos[::2]], rotation=45)
    
    # Plot 8: Standard deviation over time
    ax8 = plt.subplot(3, 3, 8)
    pic_std = np.std(pic_array, axis=1)
    mux_std = np.std(mux_array, axis=1)
    ax8.plot(x_pos, pic_std, 'o-', label='PIC MPD Std', linewidth=2, markersize=6)
    ax8.plot(x_pos, mux_std, 's-', label='MUX MPD Std', linewidth=2, markersize=6)
    ax8.set_title('Standard Deviation Over Time', fontsize=12, pad=20)
    ax8.set_xlabel('Time Sample')
    ax8.set_ylabel('Standard Deviation')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(x_pos[::2])
    ax8.set_xticklabels([timestamps[i][:8] for i in x_pos[::2]], rotation=45)
    
    # Plot 9: Channel variance comparison
    ax9 = plt.subplot(3, 3, 9)
    pic_var = np.var(pic_array, axis=0)
    mux_var = np.var(mux_array, axis=0)
    
    x = np.arange(len(channels))
    width = 0.35
    ax9.bar(x - width/2, pic_var, width, label='PIC MPD Variance', alpha=0.8)
    ax9.bar(x + width/2, mux_var, width, label='MUX MPD Variance', alpha=0.8)
    ax9.set_title('Channel Variance Comparison', fontsize=12, pad=20)
    ax9.set_xlabel('Channel')
    ax9.set_ylabel('Variance')
    ax9.set_xticks(x)
    ax9.set_xticklabels(channels)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Adjust layout with more spacing
    plt.subplots_adjust(
        left=0.05,      # Left margin
        bottom=0.08,    # Bottom margin
        right=0.95,     # Right margin
        top=0.92,       # Top margin
        wspace=0.4,     # Width spacing between subplots
        hspace=0.6      # Height spacing between subplots
    )
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("CLM LOGS SUMMARY STATISTICS")
    print("="*50)
    print(f"Data Points: {len(timestamps)} time samples, 16 channels each")
    print(f"Time Range: {timestamps[0]} to {timestamps[-1]}")
    print("\nPIC MPD Statistics:")
    print(f"  Mean: {np.mean(pic_array):.2f}")
    print(f"  Std:  {np.std(pic_array):.2f}")
    print(f"  Min:  {np.min(pic_array):.2f}")
    print(f"  Max:  {np.max(pic_array):.2f}")
    print("\nMUX MPD Statistics:")
    print(f"  Mean: {np.mean(mux_array):.2f}")
    print(f"  Std:  {np.std(mux_array):.2f}")
    print(f"  Min:  {np.min(mux_array):.2f}")
    print(f"  Max:  {np.max(mux_array):.2f}")

def plot_latest_values_only():
    """Create a single plot showing only latest values by channel"""
    # Parse the data
    timestamps, pic_data, mux_data = parse_clm_data('CLM Logs - Bench 1.csv')
    
    # Convert to numpy arrays
    pic_array = np.array(pic_data)
    mux_array = np.array(mux_data)
    
    # Create a single figure
    plt.figure(figsize=(12, 8))
    
    # Get latest values (last row of data)
    channels = list(range(16))
    latest_pic = pic_array[-1]
    latest_mux = mux_array[-1]
    
    # Create bar chart
    x = np.arange(len(channels))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, latest_pic, width, label='PIC MPD', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x + width/2, latest_mux, width, label='MUX MPD', alpha=0.8, color='lightcoral')
    
    plt.title(f'Latest Values by Channel ({timestamps[-1]})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(x, channels)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print latest values summary
    print("\n" + "="*50)
    print(f"LATEST VALUES SUMMARY - {timestamps[-1]}")
    print("="*50)
    print("Channel | PIC MPD | MUX MPD | Difference")
    print("-" * 40)
    for i in range(16):
        diff = latest_pic[i] - latest_mux[i]
        print(f"   {i:2d}   |   {latest_pic[i]:3d}   |   {latest_mux[i]:3d}   |    {diff:+3d}")
    
    print(f"\nPIC MPD - Latest Average: {np.mean(latest_pic):.1f}")
    print(f"MUX MPD - Latest Average: {np.mean(latest_mux):.1f}")
    print(f"Average Difference: {np.mean(latest_pic - latest_mux):+.1f}")

if __name__ == "__main__":
    plot_latest_values_only()  # Changed from create_dashboard()