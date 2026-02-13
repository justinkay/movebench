"""
deepSSF_functions.py
==============

Description:
    Helper functions that are used in other scripts.

Authors:
    Scott Forrest (scottwforrest@gmail.com)

Date:
    2025-02-10
    
Usage:
    Run the script as a module, such as in a Jupyter notebook or Python console:
        >>> import deepSSF_functions
        >>> deepSSF_functions.subset_raster_with_padding_torch(args)
    
    Or execute from the command line:
        $ python deepSSF_functions.py [optional arguments]
"""

# Import required libraries
import os                                       # OS - for file and directory operations
import torch                                    # PyTorch - for tensor operations and GPU support
import numpy as np                              # NumPy - for numerical operations and array handling
import glob                                     # Glob - for file path pattern matching
import imageio.v2 as imageio                    # Image manipulation - for creating GIFs
from IPython.display import Image, display      # For plotting GIFs

def subset_raster_all_bands_torch(raster_tensor, x, y, window_size, transform):
    """
    Extract windowed subset for ALL bands at once.
    
    Parameters:
        raster_tensor (torch.Tensor): 3D tensor [bands, height, width]
        x, y (float): Geographic coordinates
        window_size (int): Size of the window
        transform: Transform object
        
    Returns:
        tuple: (subset, col_start, row_start)
            subset (torch.Tensor): The windowed subset with padding. [bands, window_size, window_size]
            col_start (int): The starting column index of the window in the raster.
            row_start (int): The starting row index of the window in the raster.
    """
    # Convert GPS location to pixel coordinates
    px, py = ~transform * (x, y)
    # Floor the pixel coordinates to the get the pixel that the location is within.
    px, py = int(np.floor(px)), int(np.floor(py))

    # Compute half the window size to determine the extent around the central pixel.
    half_window = window_size // 2

    # Determine the boundaries of the window centred on the pixel coordinates.
    # Note: row_start and col_start are the top-left corner of the window.
    row_start = py - half_window
    row_stop = py + half_window + 1
    col_start = px - half_window
    col_stop = px + half_window + 1
    
    # Create output tensor for all bands
    n_bands = raster_tensor.shape[0]
    # Initialise a tensor for the subset with a padding value of -1.0.
    subset = torch.full((n_bands, window_size, window_size), -1.0, dtype=raster_tensor.dtype)
    
    # Calculate the corresponding region within the subset tensor.
    valid_row_start = max(0, row_start)
    valid_row_stop = min(raster_tensor.shape[1], row_stop)  # Note: shape[1] for height
    valid_col_start = max(0, col_start)
    valid_col_stop = min(raster_tensor.shape[2], col_stop)  # Note: shape[2] for width
    
    # Copy the valid region from the raster tensor into the appropriate section of the subset tensor.
    subset_row_start = valid_row_start - row_start
    subset_row_stop = subset_row_start + (valid_row_stop - valid_row_start)
    subset_col_start = valid_col_start - col_start
    subset_col_stop = subset_col_start + (valid_col_stop - valid_col_start)
    
    # Vectorized copy for ALL bands at once
    subset[:, subset_row_start:subset_row_stop, subset_col_start:subset_col_stop] = \
        raster_tensor[:, valid_row_start:valid_row_stop, valid_col_start:valid_col_stop]
    
    # Return the subset along with the starting column and row indices of the window.
    return subset, col_start, row_start


def subset_raster_with_padding_torch(raster_tensor, x, y, window_size, transform):
    """
    Extracts a windowed subset from a raster tensor centered at the geographic coordinate (x, y).
    Areas outside the raster are padded with -1.0.
    
    Parameters:
        raster_tensor (torch.Tensor): 2D tensor containing raster data.
        x, y (float): Geographic coordinates.
        window_size (int): Size of the window (typically an odd number).
        transform: An object supporting inversion and multiplication to map geographic to pixel coordinates.
        
    Returns:
        tuple: (subset, col_start, row_start)
            subset (torch.Tensor): The windowed subset with padding.
            col_start (int): The starting column index of the window in the raster.
            row_start (int): The starting row index of the window in the raster.
    """
        
    # Convert geographic coordinates to pixel coordinates using the inverse transform.
    px, py = ~transform * (x, y)
    
    # Floor the pixel coordinates to the get the pixel that the location is within.
    px, py = int(np.floor(px)), int(np.floor(py))
    
    # Compute half the window size to determine the extent around the central pixel.
    half_window = window_size // 2
    
    # Determine the boundaries of the window centred on the pixel coordinates.
    row_start = py - half_window
    row_stop = py + half_window + 1
    col_start = px - half_window
    col_stop = px + half_window + 1
    
    # Initialise a tensor for the subset with a padding value of -1.0.
    subset = torch.full((window_size, window_size), -1.0, dtype=raster_tensor.dtype)
    
    # Determine the valid region of the raster that falls within the window boundaries.
    valid_row_start = max(0, row_start)
    valid_row_stop = min(raster_tensor.shape[0], row_stop)
    valid_col_start = max(0, col_start)
    valid_col_stop = min(raster_tensor.shape[1], col_stop)
    
    # Calculate the corresponding region within the subset tensor.
    subset_row_start = valid_row_start - row_start
    subset_row_stop = subset_row_start + (valid_row_stop - valid_row_start)
    subset_col_start = valid_col_start - col_start
    subset_col_stop = subset_col_start + (valid_col_stop - valid_col_start)
    
    # Copy the valid region from the raster tensor into the appropriate section of the subset tensor.
    subset[subset_row_start:subset_row_stop, subset_col_start:subset_col_stop] = \
        raster_tensor[valid_row_start:valid_row_stop, valid_col_start:valid_col_stop]
    
    # Return the subset along with the starting column and row indices of the window.
    return subset, col_start, row_start


def subset_raster_with_padding_npy(raster_npy, x, y, window_size, transform):
    """
    Extracts a windowed subset from a raster tensor centered at the geographic coordinate (x, y).
    Areas outside the raster are padded with -1.0.
    
    Parameters:
        raster as npy: 2D array containing raster data.
        x, y (float): Geographic coordinates.
        window_size (int): Size of the window (typically an odd number).
        transform: An object supporting inversion and multiplication to map geographic to pixel coordinates.
        
    Returns:
        tuple: (subset, col_start, row_start)
            subset (np.array): The windowed subset with padding.
            col_start (int): The starting column index of the window in the raster.
            row_start (int): The starting row index of the window in the raster.
    """
        
    # Convert geographic coordinates to pixel coordinates using the inverse transform.
    px, py = ~transform * (x, y)
    
    # Floor the pixel coordinates to the get the pixel that the location is within.
    px, py = int(np.floor(px)), int(np.floor(py))
    
    # Compute half the window size to determine the extent around the central pixel.
    half_window = window_size // 2
    
    # Determine the boundaries of the window centred on the pixel coordinates.
    row_start = py - half_window
    row_stop = py + half_window + 1
    col_start = px - half_window
    col_stop = px + half_window + 1
    
    # Initialise a tensor for the subset with a padding value of -1.0.
    # subset = torch.full((window_size, window_size), -1.0, dtype=raster_npy.dtype)

    # Initialise a numpy array for the subset with a padding value of -1.0.
    subset = np.full((window_size, window_size), -1.0, dtype=raster_npy.dtype)
    
    # Determine the valid region of the raster that falls within the window boundaries.
    valid_row_start = max(0, row_start)
    valid_row_stop = min(raster_npy.shape[0], row_stop)
    valid_col_start = max(0, col_start)
    valid_col_stop = min(raster_npy.shape[1], col_stop)
    
    # Calculate the corresponding region within the subset tensor.
    subset_row_start = valid_row_start - row_start
    subset_row_stop = subset_row_start + (valid_row_stop - valid_row_start)
    subset_col_start = valid_col_start - col_start
    subset_col_stop = subset_col_start + (valid_col_stop - valid_col_start)
    
    # Copy the valid region from the raster tensor into the appropriate section of the subset tensor.
    subset[subset_row_start:subset_row_stop, subset_col_start:subset_col_stop] = \
        raster_npy[valid_row_start:valid_row_stop, valid_col_start:valid_col_stop]
    
    # Return the subset along with the starting column and row indices of the window.
    return subset, col_start, row_start



def recover_hour(sin_term, cos_term):
    """
    Recovers the hour of the day from its sine and cosine encoding.

    Parameters:
        sin_term (float or np.array): Sine-transformed hour value.
        cos_term (float or np.array): Cosine-transformed hour value.

    Returns:
        float or np.array: Recovered hour in the range [0, 24).
    """
    # Calculate the angle theta
    theta = np.arctan2(sin_term, cos_term)
    # Convert to hour
    hour = (24 * theta) / (2 * np.pi) % 24
    return hour



def recover_yday(sin_term, cos_term):
    """
    Recovers the day of the year from its sine and cosine encoding.

    Parameters:
        sin_term (float or np.array): Sine-transformed day value.
        cos_term (float or np.array): Cosine-transformed day value.

    Returns:
        float or np.array: Recovered day of the year in the range [0, 365).
    """
    # Calculate the angle theta
    theta = np.arctan2(sin_term, cos_term)
    # Convert to day of year
    yday = (365.25 * theta) / (2 * np.pi) % 365.25
    return yday


# To clear the GPU memory cache
def clear_memory(device):
    # Clear cache based on device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.empty_cache()
    # CPU doesn't need explicit cache clearing


# Example sorting by the epoch number
def extract_index(filename):
    # Extract the epoch number from the filename
    # Adjust the extraction based on your naming pattern
    import re
    match = re.search(r'index(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0

def create_gif(image_folder, output_filename, fps=10):
    """
    Creates a GIF from a sequence of images in a folder.

    Parameters:
    - image_folder: Path to the folder containing images
    - output_filename: Name of the output GIF file
    - duration: Duration of each frame in seconds
    """
    # Get all png files in the specified folder, sorted by name
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=extract_index)

    # Check if any images were found
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Read all images
    frames = [imageio.imread(image) for image in images]

    # Save as GIF
    imageio.mimsave(output_filename, frames, fps=fps, loop=0)

    display(Image(filename=output_filename))

    print(f"GIF created successfully: {output_filename}")