import os
import tempfile
import glob
from typing import Tuple, List


def find_bins_by_type(data_dir: str, bin_type: str) -> List[str]:
    """Find all bins of the specified type (I or D) in the data directory.
    
    Args:
        data_dir: Directory containing IFCB point cloud data
        bin_type: Type of bins to find ('I' or 'D')
        
    Returns:
        List of PIDs (without .adc extension) matching the bin type
    """
    # Find all .adc files in the data directory
    adc_files = glob.glob(os.path.join(data_dir, "**", "*.adc"), recursive=True)
    
    # Extract PIDs and filter by bin type
    filtered_pids = []
    for adc_file in adc_files:
        filename = os.path.basename(adc_file)
        pid = os.path.splitext(filename)[0]  # Remove .adc extension
        
        if pid.startswith(bin_type):
            filtered_pids.append(pid)
    
    return filtered_pids


def create_bin_type_id_file(data_dir: str, bin_type: str) -> Tuple[str, int]:
    """Create a temporary ID file containing only bins of the specified type (I or D).
    
    Args:
        data_dir: Directory containing IFCB point cloud data
        bin_type: Type of bins to include ('I' or 'D')
        
    Returns:
        Tuple of (temp_file_path, number_of_bins_found)
    """
    filtered_pids = find_bins_by_type(data_dir, bin_type)
    
    if not filtered_pids:
        return None, 0
    
    # Create temporary ID file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix=f'{bin_type}_bins_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            for pid in filtered_pids:
                f.write(f"{pid}\n")
    except:
        os.unlink(temp_path)
        raise
    
    return temp_path, len(filtered_pids)