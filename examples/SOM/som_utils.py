import os
import rasterio
import numpy as np

def combine_tifs(directory):
    data = []
    count = 0
    print(f"Preparing to combine .tif files from directory {directory}.")
    # Read each .tif file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            print(filename)
            with rasterio.open(os.path.join(directory, filename)) as src:
                # Read the data and flatten it into a 1D array
                data.append(src.read(1).flatten())
                count += 1
    
    # Convert the list to a numpy array
    data = np.array(data)
    print(f"Combined data from {count} .tif files.")
    return data


def add_tifs(directory):
    # count = 0
    # print(f"Preparing to add together .tif files from directory {directory}.")
    # for filename in os.listdir(directory):
    #     if filename.endswith('.tif'):
    #         print(filename)
        
    #         with rasterio.open(os.path.join(directory, filename)) as src:
    #             if count == 0:
    #                 data = src.read(1).flatten()
    #             else:
    #                 data = np.add(data, src.read(1).flatten())
    #             count += 1
    # print(f"Combined data from {count} .tif files.")
    # return data
    print("addfunc")
    return "data"