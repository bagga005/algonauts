import os
import utils
import gzip
import pickle
def save_embeddings(full_embeddings, predictions, prefix, counter):
    save_dir = os.path.join(utils.get_output_dir(), utils.get_embeddings_dir())
    print(f'save_dir: {save_dir}')
    assert full_embeddings.shape[0] == predictions.shape[0], 'full_embeddings and predictions must have the same number of rows'
    start_index = counter
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Create a metadata dictionary to store shapes and types
    metadata = {}
    file_ext = ".pt.gz"
    
    layer_name = 'blocks.5.pool'
    safe_name = layer_name.replace('.', '_').replace('/', '_')
    base_dir = os.path.join(save_dir, safe_name)
    os.makedirs(base_dir, exist_ok=True)
    for i in range(start_index, start_index + full_embeddings.shape[0]):
        prefix_with_counter = f"{prefix}_tr_{i}"
        safe_name = f"{prefix_with_counter}_{safe_name}"
        
        embeddings = predictions[i,:]
        with gzip.open(os.path.join(base_dir, safe_name + file_ext), 'wb') as f:
            pickle.dump(embeddings, f)
        metadata[layer_name] = {
                    'type': 'numpy.ndarray',
                    'shape': list(embeddings.shape) if hasattr(embeddings, 'shape') else None
        }
    
    layer_name = 'predictions'
    safe_name = layer_name.replace('.', '_').replace('/', '_')
    base_dir = os.path.join(save_dir, safe_name)
    os.makedirs(base_dir, exist_ok=True)
    for i in range(start_index, start_index + predictions.shape[0]):
        prefix_with_counter = f"{prefix}_tr_{i}"
        safe_name = f"{prefix_with_counter}_{safe_name}"
        
        embeddings = predictions[i,:]
        print(base_dir, safe_name + file_ext)
        with gzip.open(os.path.join(base_dir, safe_name + file_ext), 'wb') as f:
            pickle.dump(embeddings, f)
        metadata[layer_name] = {
                    'type': 'np',
                    'shape': list(embedding.shape) if hasattr(embedding, 'shape') else None
        }
    
    
    
    utils.save_embedding_metadata(prefix_with_counter, metadata)
    return counter + predictions.shape[0]