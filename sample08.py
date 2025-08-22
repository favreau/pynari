#!/usr/bin/env python3
"""
Sample08.py - Volume Sequence Renderer

This script renders a sequence of volume files from an input folder to an output folder.
Each volume file is rendered as a PNG image with the same name but .png extension.

Supported formats:
    - .mhd files (MetaImage format with header and data)
    - .raw files (raw binary float32 data)
    - .npy files (numpy arrays)
    - Other binary files (auto-detected as float32 data)

Usage:
    python sample08.py --input /path/to/volume/files --output /path/to/output
    python sample08.py -i /path/to/volume/files -o /path/to/output

Requirements:
    - pynari (ANARI rendering with barney backend)
    - numpy
    - PIL (Pillow)
"""

import os
import sys
import argparse
import numpy as np
import pynari as anari
from PIL import Image
import glob
from pathlib import Path
import time



def load_volume_file(volume_path):
    """
    Load a volume file and extract volume data.
    Supports various formats including raw files, numpy arrays, and MHD files.
    """
    volume_path = Path(volume_path)
    
    if volume_path.suffix.lower() == '.mhd':
        # Load MHD (MetaImage) file
        try:
            print(f"Loading MHD file {volume_path}")
            return load_mhd_file(volume_path)
        except Exception as e:
            print(f"Error loading MHD file {volume_path}: {e}")
            return create_placeholder_volume(), (1.0, 1.0, 1.0)
    
    elif volume_path.suffix.lower() == '.raw':
        # Load raw volume file (like in sample03.py)
        try:
            # Try common dimensions first
            for dims in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)]:
                try:
                    data = np.fromfile(volume_path, dtype=np.float32)
                    if data.size == dims[0] * dims[1] * dims[2]:
                        print(f"Loading raw file {volume_path} with dimensions {dims}")
                        volume_data = data.reshape(dims)
                        return volume_data, (2.0/dims[0], 2.0/dims[1], 2.0/dims[2])
                except:
                    continue
            
            # If no standard dimensions work, try to infer from file size
            data = np.fromfile(volume_path, dtype=np.float32)
            size = len(data)
            # Try to find reasonable cube dimensions
            for i in range(1, int(size**(1/3)) + 1):
                if size % (i**3) == 0:
                    dim = i
                    if dim**3 == size:
                        volume_data = data.reshape((dim, dim, dim))
                        return volume_data, (2.0/dim, 2.0/dim, 2.0/dim)
            
            # Fallback: use as 1D and reshape to cube
            dim = int(size**(1/3))
            volume_data = data[:dim**3].reshape((dim, dim, dim))
            return volume_data, (2.0/dim, 2.0/dim, 2.0/dim)
            
        except Exception as e:
            print(f"Error loading raw file {volume_path}: {e}")
            return create_placeholder_volume(), (1.0, 1.0, 1.0)
    
    elif volume_path.suffix.lower() == '.npy':
        # Load numpy array file
        try:
            volume_data = np.load(volume_path)
            if volume_data.ndim == 3:
                dims = volume_data.shape
                return volume_data, (2.0/dims[0], 2.0/dims[1], 2.0/dims[2])
            else:
                print(f"Error: {volume_path} is not a 3D array")
                return create_placeholder_volume(), (1.0, 1.0, 1.0)
        except Exception as e:
            print(f"Error loading numpy file {volume_path}: {e}")
            return create_placeholder_volume(), (1.0, 1.0, 1.0)
    
    else:
        # For other formats, try to create a simple volume based on file size
        try:
            file_size = volume_path.stat().st_size
            # Assume float32 data
            num_floats = file_size // 4
            dim = int(num_floats**(1/3))
            if dim**3 == num_floats:
                data = np.fromfile(volume_path, dtype=np.float32)
                volume_data = data.reshape((dim, dim, dim))
                return volume_data, (2.0/dim, 2.0/dim, 2.0/dim)
        except:
            pass
        
        # Fallback to placeholder
        print(f"Unsupported file format: {volume_path.suffix}")
        return create_placeholder_volume(), (1.0, 1.0, 1.0)

def create_placeholder_volume():
    """Create a placeholder volume for testing when VDB files are not available."""
    # Create a simple sphere-like volume
    size = 64
    volume = np.zeros((size, size, size), dtype=np.float32)
    
    center = size // 2
    radius = size // 3
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                if dist < radius:
                    volume[i, j, k] = max(0, 1.0 - dist / radius)
    
    return volume

def load_mhd_file(mhd_path):
    """
    Load an MHD (MetaImage) file and extract volume data.
    MHD files contain header information and reference to raw data files.
    """
    mhd_path = Path(mhd_path)
    
    # Read MHD header
    header = {}
    with open(mhd_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    header[key.strip()] = value.strip()
    
    # Extract key information
    dims = [int(x) for x in header.get('DimSize', '64 64 64').split()]
    spacing = [float(x) for x in header.get('ElementSpacing', '1.0 1.0 1.0').split()]
    offset = [float(x) for x in header.get('Offset', '0 0 0').split()]
    data_type = header.get('ElementType', 'MET_FLOAT')
    data_file = header.get('ElementDataFile', '')
    
    # Map MHD data types to numpy types
    type_mapping = {
        'MET_UCHAR': np.uint8,
        'MET_CHAR': np.int8,
        'MET_USHORT': np.uint16,
        'MET_SHORT': np.int16,
        'MET_UINT': np.uint32,
        'MET_INT': np.int32,
        'MET_FLOAT': np.float32,
        'MET_DOUBLE': np.float64
    }
    
    numpy_type = type_mapping.get(data_type, np.float32)
    
    # Determine data file path
    if data_file:
        if Path(data_file).is_absolute():
            data_path = Path(data_file)
        else:
            data_path = mhd_path.parent / data_file
    else:
        # If no data file specified, assume raw data follows header
        data_path = mhd_path
    
    # Load volume data
    try:
        if data_path == mhd_path:
            # Data is embedded in MHD file (rare)
            with open(mhd_path, 'rb') as f:
                # Skip header (find double newline)
                content = f.read()
                data_start = content.find(b'\n\n')
                if data_start == -1:
                    raise ValueError("Could not find data section in MHD file")
                
                f.seek(data_start + 2)
                volume_data = np.fromfile(f, dtype=numpy_type)
        else:
            # Data is in separate file
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            volume_data = np.fromfile(data_path, dtype=numpy_type)
        
        # Reshape to 3D
        if len(dims) == 3:
            volume_data = volume_data.reshape(dims, order='F')  # Fortran order for MHD
        else:
            # Handle 2D or 1D data
            if len(dims) == 2:
                volume_data = volume_data.reshape(dims + [1], order='F')
            else:
                # Assume cubic volume
                size = int(len(volume_data) ** (1/3))
                volume_data = volume_data[:size**3].reshape((size, size, size))
        
        # Normalize to 0-1 range if needed
        if volume_data.dtype in [np.uint8, np.uint16, np.uint32]:
            volume_data = volume_data.astype(np.float32) / volume_data.max()
        
        # Calculate voxel spacing and origin
        if len(spacing) >= 3:
            voxel_spacing = (spacing[0], spacing[1], spacing[2])
        else:
            # Default spacing
            voxel_spacing = (2.0/dims[0], 2.0/dims[1], 2.0/dims[2])
        
        return volume_data, voxel_spacing, offset
        
    except Exception as e:
        print(f"Error loading MHD data: {e}")
        return create_placeholder_volume(), (1.0, 1.0, 1.0), [0, 0, 0]

def setup_camera(device, width, height, volume_data=None, voxel_size=None, offset=None):
    """Setup camera for volume rendering."""
    camera = device.newCamera('perspective')
    camera.setParameter('aspect', anari.FLOAT32, width / height)
    
    # Calculate optimal camera position based on volume dimensions, spacing, and offset
    if volume_data is not None and voxel_size is not None and offset is not None:
        dims = volume_data.shape
        
        # Calculate volume bounds
        min_x = offset[0]
        max_x = offset[0] + dims[0] * voxel_size[0]
        min_y = offset[1]
        max_y = offset[1] + dims[1] * voxel_size[1]
        min_z = offset[2]
        max_z = offset[2] + dims[2] * voxel_size[2]
        
        # Calculate volume center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        # Calculate volume size (diagonal)
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        volume_diagonal = (size_x**2 + size_y**2 + size_z**2)**0.5
        
        # Position camera at a distance proportional to volume size
        # Use 1.5x the volume diagonal to ensure the entire volume is visible
        camera_distance = volume_diagonal * 1.5
        
        # Position camera above and slightly to the side for better perspective
        camera_x = center_x + camera_distance * 0.3
        camera_y = center_y + camera_distance * 0.3
        camera_z = center_z + camera_distance * 0.8
        
        # Calculate direction vector from camera to volume center
        direction_x = center_x - camera_x
        direction_y = center_y - camera_y
        direction_z = center_z - camera_z
        
        # Normalize direction vector
        direction_length = (direction_x**2 + direction_y**2 + direction_z**2)**0.5
        direction_x /= direction_length
        direction_y /= direction_length
        direction_z /= direction_length
        
        camera.setParameter('position', anari.FLOAT32_VEC3, [camera_x, camera_y, camera_z])
        camera.setParameter('direction', anari.float3, [direction_x, direction_y, direction_z])
        
        # Debug output for camera positioning
        print(f"  Volume bounds: X[{min_x:.1f}, {max_x:.1f}] Y[{min_y:.1f}, {max_y:.1f}] Z[{min_z:.1f}, {max_z:.1f}]")
        print(f"  Volume center: [{center_x:.1f}, {center_y:.1f}, {center_z:.1f}]")
        print(f"  Camera position: [{camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f}]")
        print(f"  Camera distance: {camera_distance:.1f}")
        
    else:
        # Fallback to default positioning
        camera.setParameter('position', anari.FLOAT32_VEC3, [0, 0, 5])
        camera.setParameter('direction', anari.float3, [0, 0, -1])
    camera.setParameter('up', anari.float3, [0, 1, 0])
    camera.setParameter('fovy', anari.FLOAT32, 45.0 * 3.14159 / 180.0)
    camera.commitParameters()
    return camera

def setup_renderer(device):
    """Setup renderer for volume rendering."""
    renderer = device.newRenderer('default')
    renderer.setParameter('pixelSamples', anari.INT32, 256)
    renderer.setParameter('ambientRadiance', anari.FLOAT32, 1.0)
    
    # Background gradient
    # bg_values = np.array(((.1, .1, .1, 1.), (.1, .1, .1, 1.)), dtype=np.float32).reshape((4, 1, 2))
    bg_values = np.array(((.0, .0, .0, .0), (.0, .0, .0, .0)), dtype=np.float32).reshape((4, 1, 2))
    bg_gradient = device.newArray(anari.float4, bg_values)
    renderer.setParameter('background', anari.ARRAY, bg_gradient)
    
    renderer.commitParameters()
    return renderer

def setup_lighting(device):
    """Setup lighting for the scene."""
    light = device.newLight('directional')
    light.setParameter('direction', anari.float3, [1.0, -1.0, -1.0])
    light.setParameter('irradiance', anari.float, 2.0)  # Brighter light for bounding box visibility
    light.commitParameters()
    
    array = device.newArray(anari.LIGHT, [light])
    return array

def create_bounding_box(device, volume_data, voxel_size, offset=[0, 0, 0]):
    """Create a wireframe bounding box around the volume."""
    dims = volume_data.shape
    
    # Calculate bounding box corners (slightly larger than volume for visibility)
    if offset != [0, 0, 0]:
        min_corner = [offset[0] - 10, offset[1] - 10, offset[2] - 10]  # Slightly larger
        max_corner = [offset[0] + dims[0] * voxel_size[0] + 10, 
                     offset[1] + dims[1] * voxel_size[1] + 10, 
                     offset[2] + dims[2] * voxel_size[2] + 10]
    else:
        min_corner = [-dims[0] * voxel_size[0] / 2 - 10, -dims[1] * voxel_size[1] / 2 - 10, -dims[2] * voxel_size[2] / 2 - 10]
        max_corner = [dims[0] * voxel_size[0] / 2 + 10, dims[1] * voxel_size[1] / 2 + 10, dims[2] * voxel_size[2] / 2 + 10]
    
    # Create wireframe box using curves
    box_geometry = device.newGeometry('curve')
    
    # Define the 12 edges of the bounding box
    edges = []
    # Bottom face
    edges.extend([min_corner[0], min_corner[1], min_corner[2]])
    edges.extend([max_corner[0], min_corner[1], min_corner[2]])
    edges.extend([max_corner[0], min_corner[1], min_corner[2]])
    edges.extend([max_corner[0], max_corner[1], min_corner[2]])
    edges.extend([max_corner[0], max_corner[1], min_corner[2]])
    edges.extend([min_corner[0], max_corner[1], min_corner[2]])
    edges.extend([min_corner[0], max_corner[1], min_corner[2]])
    edges.extend([min_corner[0], min_corner[1], min_corner[2]])
    
    # Top face
    edges.extend([min_corner[0], min_corner[1], max_corner[2]])
    edges.extend([max_corner[0], min_corner[1], max_corner[2]])
    edges.extend([max_corner[0], min_corner[1], max_corner[2]])
    edges.extend([max_corner[0], max_corner[1], max_corner[2]])
    edges.extend([max_corner[0], max_corner[1], max_corner[2]])
    edges.extend([min_corner[0], max_corner[1], max_corner[2]])
    edges.extend([min_corner[0], max_corner[1], max_corner[2]])
    edges.extend([min_corner[0], min_corner[1], max_corner[2]])
    
    # Vertical edges
    edges.extend([min_corner[0], min_corner[1], min_corner[2]])
    edges.extend([min_corner[0], min_corner[1], max_corner[2]])
    edges.extend([max_corner[0], min_corner[1], min_corner[2]])
    edges.extend([max_corner[0], min_corner[1], max_corner[2]])
    edges.extend([max_corner[0], max_corner[1], min_corner[2]])
    edges.extend([max_corner[0], max_corner[1], max_corner[2]])
    edges.extend([min_corner[0], max_corner[1], min_corner[2]])
    edges.extend([min_corner[0], max_corner[1], max_corner[2]])
    
    # Create vertex positions array
    vertex_positions = np.array(edges, dtype=np.float32)
    vertex_array = device.newArray(anari.FLOAT32_VEC3, vertex_positions)
    box_geometry.setParameter('vertex.position', anari.ARRAY, vertex_array)
    
    # Create vertex colors (bright green for visibility)
    vertex_colors = []
    for i in range(len(edges) // 3):
        vertex_colors.extend([0.0, 1.0, 0.0, 1.0])  # Bright green
    vertex_colors = np.array(vertex_colors, dtype=np.float32)
    color_array = device.newArray(anari.FLOAT32_VEC4, vertex_colors)
    box_geometry.setParameter('vertex.color', anari.ARRAY, color_array)
    
    # Create vertex radius (very thick lines for visibility)
    vertex_radius = np.array([5.0] * (len(edges) // 3), dtype=np.float32)  # Very thick
    radius_array = device.newArray(anari.FLOAT32, vertex_radius)
    box_geometry.setParameter('vertex.radius', anari.ARRAY, radius_array)
    
    # Create indices for line segments
    indices = []
    for i in range(0, len(edges) // 3, 2):
        indices.extend([i, i + 1])
    
    indices = np.array(indices, dtype=np.uint32)
    index_array = device.newArray(anari.UINT32, indices)
    box_geometry.setParameter('primitive.index', anari.ARRAY, index_array)
    
    box_geometry.commitParameters()
    
    # Create surface for the bounding box
    box_surface = device.newSurface()
    box_surface.setParameter('geometry', anari.GEOMETRY, box_geometry)
    
    # Create material for wireframe
    material = device.newMaterial('matte')
    material.setParameter('color', anari.float3, [0.0, 1.0, 0.0])  # Bright green
    material.commitParameters()
    
    box_surface.setParameter('material', anari.MATERIAL, material)
    box_surface.commitParameters()
    
    return box_surface

def create_volume_from_data(device, volume_data, voxel_size, offset=[0, 0, 0]):
    """Create an ANARI volume from volume data."""
    # Create spatial field
    spatial_field = device.newSpatialField('structuredRegular')
    
    # For medical volumes, use the actual volume bounds with proper offset
    dims = volume_data.shape
    if offset != [0, 0, 0]:
        # Use the exact offset from MHD file
        origin = offset
    else:
        # Calculate center-based origin
        origin = [-dims[0] * voxel_size[0] / 2, -dims[1] * voxel_size[1] / 2, -dims[2] * voxel_size[2] / 2]
    spatial_field.setParameter('origin', anari.float3, origin)
    spatial_field.setParameter('spacing', anari.float3, voxel_size)
    
    # Create array from volume data
    volume_array = device.newArray(anari.float, volume_data)
    spatial_field.setParameter('data', anari.ARRAY3D, volume_array)
    spatial_field.commitParameters()
    
    # Create transfer function for 0-1 range data
    # Generate 256 samples for smooth mapping
    # Make even very low values visible since volume mean is 0.002
    tf_data = []
    for i in range(256):
        value = i / 255.0
        
        # Map very low values (0-0.01) to transparent
        if value < 0.01:
            opacity = 0.0
            color = [0, 0, 0]
        # Map low values (0.01-0.1) to blue with high opacity
        elif value < 0.1:
            opacity = 0.5 + (value - 0.01) / 0.09 * 0.5  # 0.5 to 1.0 opacity
            color = [0, 0, 1]  # Blue
        # Map medium values (0.1-0.5) to green
        elif value < 0.5:
            opacity = 1.0
            color = [0, 1, 0]  # Green
        # Map high values (0.5-1.0) to red
        else:
            opacity = 1.0
            color = [1, 0, 0]  # Red
        
        tf_data.extend([color[0], color[1], color[2], opacity])
    
    tf_data = np.array(tf_data, dtype=np.float32)
    
    tf_array = device.newArray(anari.float4, tf_data)
    
    # Create volume
    volume = device.newVolume('transferFunction1D')
    volume.setParameter('color', anari.ARRAY, tf_array)
    volume.setParameter('value', anari.SPATIAL_FIELD, spatial_field)
    volume.setParameter('unitDistance', anari.FLOAT32, 10.0)
    volume.commitParameters()
    
    return volume

def render_volume_file(device, volume_path, output_path, width=1024, height=768, verbose=False, show_bbox=False):
    """Render a single volume file to PNG."""
    start_time = time.time()
    print(f"Rendering {volume_path} -> {output_path}")
    
    # Load volume data
    if verbose:
        print(f"  Loading volume data from {volume_path}")
    load_start = time.time()
    result = load_volume_file(volume_path)
    if verbose:
        load_time = time.time() - load_start
        print(f"  Volume data loaded in {load_time:.3f}s")
    
    if len(result) == 3:
        volume_data, voxel_size, offset = result
        if verbose:
            print(f"  Volume dimensions: {volume_data.shape}")
            print(f"  Voxel spacing: {voxel_size}")
            print(f"  Volume offset: {offset}")
            print(f"  Volume data range: [{volume_data.min():.3f}, {volume_data.max():.3f}]")
            print(f"  Volume data mean: {volume_data.mean():.3f}")
            print(f"  Volume data std: {volume_data.std():.3f}")
    else:
        volume_data, voxel_size = result
        offset = [0, 0, 0]
        if verbose:
            print(f"  Volume dimensions: {volume_data.shape}")
            print(f"  Voxel spacing: {voxel_size}")
            print(f"  No offset specified")
    
    # Setup scene
    if verbose:
        print(f"  Creating world and scene objects")
    world = device.newWorld()
    
    # Create volume
    if verbose:
        print(f"  Creating volume from data")
    volume = create_volume_from_data(device, volume_data, voxel_size, offset)
    world.setParameterArray('volume', anari.VOLUME, [volume])
    
    # Add bounding box if requested
    surfaces = []
    if show_bbox:
        if verbose:
            print(f"  Creating bounding box")
        bbox_surface = create_bounding_box(device, volume_data, voxel_size, offset)
        surfaces.append(bbox_surface)
    
    if surfaces:
        world.setParameterArray('surface', anari.SURFACE, surfaces)
    
    # Setup lighting
    if verbose:
        print(f"  Setting up lighting")
    light_array = setup_lighting(device)
    world.setParameter('light', anari.ARRAY1D, light_array)
    world.commitParameters()
    
    # Setup camera
    if verbose:
        print(f"  Setting up camera ({width}x{height})")
    camera = setup_camera(device, width, height, volume_data, voxel_size, offset)
    
    # Setup renderer
    if verbose:
        print(f"  Setting up renderer")
    renderer = setup_renderer(device)
    
    # Create frame
    if verbose:
        print(f"  Creating frame buffer")
    frame = device.newFrame()
    frame.setParameter('size', anari.uint2, [width, height])
    frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_VEC4)
    frame.setParameter('renderer', anari.OBJECT, renderer)
    frame.setParameter('camera', anari.OBJECT, camera)
    frame.setParameter('world', anari.OBJECT, world)
    frame.commitParameters()
    
    # Render
    if verbose:
        print(f"  Starting render...")
    render_start = time.time()
    frame.render()
    render_time = time.time() - render_start
    if verbose:
        print(f"  Render complete in {render_time:.3f}s, retrieving frame buffer")
    fb_color = frame.get('channel.color')
    pixels = np.array(fb_color)
    
    # Save image
    if verbose:
        print(f"  Saving image to {output_path}")
    save_start = time.time()
    im = Image.fromarray(pixels)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.convert('RGBA')
    im.save(output_path)
    save_time = time.time() - save_start
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  Image saved in {save_time:.3f}s")
        print(f"  Total render time: {total_time:.3f}s")
    
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Render volume files to PNG images')
    parser.add_argument('-i', '--input', required=True, help='Input folder containing volume files')
    parser.add_argument('-o', '--output', required=True, help='Output folder for PNG images')
    parser.add_argument('--width', type=int, default=1024, help='Image width (default: 1024)')
    parser.add_argument('--height', type=int, default=768, help='Image height (default: 768)')
    parser.add_argument('--pattern', default='*.raw', help='File pattern to match (default: *.raw)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging from ANARI device')
    parser.add_argument('--device', default='default', help='ANARI device to use (default: default, options: default, barney, helide)')
    parser.add_argument('--show-bbox', action='store_true', help='Render volume bounding box for debugging')
    
    args = parser.parse_args()
    
    # Validate input and output directories
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find volume files
    volume_files = list(input_dir.glob(args.pattern))
    if not volume_files:
        print(f"No files matching pattern '{args.pattern}' found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(volume_files)} files to render")
    
    # Create device with optional verbose logging
    if args.verbose:
        print(f"Initializing ANARI device: '{args.device}'...")
    device = anari.newDevice(args.device)
    
    # Enable verbose logging if requested
    if args.verbose:
        print("Enabling verbose ANARI device logging...")
        # Set device parameters for verbose output (using INT32 for boolean values)
        try:
            device.setParameter('verbose', anari.INT32, 1)
            device.setParameter('debugMode', anari.INT32, 1)
            device.commitParameters()
        except:
            # If device doesn't support these parameters, continue without them
            pass
    
    # Render each file
    total_start_time = time.time()
    successful_renders = 0
    failed_renders = 0
    
    for volume_file in sorted(volume_files):
        # Create output filename
        output_file = output_dir / f"{volume_file.stem}.png"
        
        try:
            render_volume_file(device, str(volume_file), str(output_file), args.width, args.height, args.verbose, args.show_bbox)
            successful_renders += 1
        except Exception as e:
            print(f"Error rendering {volume_file}: {e}")
            failed_renders += 1
            continue
    
    total_time = time.time() - total_start_time
    
    if args.verbose:
        print(f"\nRendering Summary:")
        print(f"  Total files processed: {len(volume_files)}")
        print(f"  Successful renders: {successful_renders}")
        print(f"  Failed renders: {failed_renders}")
        print(f"  Total time: {total_time:.3f}s")
        if successful_renders > 0:
            print(f"  Average time per render: {total_time/successful_renders:.3f}s")
    
    print("Rendering complete!")

if __name__ == "__main__":
    main()
