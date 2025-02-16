import numpy as np
import os
import time
import random
# print(os.cpu_count())
from lightning.data import map
import logging
import json_log_formatter



logger = logging.getLogger('my_json')
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.FileHandler(filename='run_log.json')
json_handler.setFormatter(formatter)
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)
# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def create_file_with_square(stimId, stim_file):
    st = random.randint(1, 10)
    time.sleep(st)
    sst = str(st)
    logger.info(stimId['brand'], extra={'sleep_time': sst})
    # logger.debug('This is my sleep time for {index} :'+  str(st))
     #output_filepath = os.path.join(output_dir, f"{index}.txt")
    # with open(stim_file, "w") as f:
     #    f.write(str(st))

if __name__ == '__main__':
    
    
    #tb_logger.log('a')
    #tb_logger.summary.text("first_text", "hello world", step=0)
    thisdict = [{
        "brand": "Ford",
        "model": "Mustang",
        "year": 1964
    },
    {
        "brand": "Ford",
        "model": "Mustang",
        "year": 1964
    }]
    map(
        fn=create_file_with_square,
        inputs=thisdict,
        num_workers=2,
        output_dir="thisdic"
    )


# # Example of saving a numpy array
# example_array = np.array([[1, 2, 3], [4, 5, 6]])
# dirpath = "/teamspace/studios/productive-tomato-3ymm/data/ann_brain_data/"  # Replace with the actual path to your .npy file
# filepath = dirpath + "my_data.npy"
# #np.save(filepath, example_array) # Saves the example array to 'my_data.npy'

# def load_and_read_npy(filepath):
#     """Loads a .npy file and returns its contents.

#     Args:
#         filepath: The path to the .npy file.

#     Returns:
#         The contents of the .npy file as a NumPy array, or None if an error occurs.
#         Prints an error message to the console if the file is not found or other issues arise.
#     """
#     try:
#         data = np.load(filepath, allow_pickle=1).item()
#         return data
#     except FileNotFoundError:
#         print(f"Error: File not found at {filepath}")
#         return None
#     except Exception as e:  # Catch other potential errors (e.g., corrupted file)
#         print(f"Error loading .npy file: {e}")
#         return None


# # Example usage:
# filepath = "/teamspace/studios/productive-tomato-3ymm/data/ann_brain_data/activations/actv-clipvit-base-patch32-bourne01-eqsTR1.49s-5layers_clsEmbd.npy"  # Replace with the actual path to your .npy file
# data = load_and_read_npy(filepath)
# print(data.keys())
# data = data[ 'vision_model.encoder.layers.11']
# if data is not None:
#     print("Data loaded successfully:")
#     print(data)  # Print the data (or process it as needed)
#     print(f"Shape of Data: {data.shape}") # Print the shape of the data
#     print(f"Data Type: {data.dtype}") # Print the data type

    

# else:
#     print("Failed to load data.")



