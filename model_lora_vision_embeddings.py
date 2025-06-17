import os
import utils
import gzip
import pickle
def save_embeddings(full_embeddings, predictions, save_dir, prefix, counter):
    """
    Save embeddings to files.
    
    Args:
        embeddings: Dictionary of layer names to embeddings
        save_dir: Directory to save the embeddings
        prefix: Optional prefix for the filenames (e.g., image name or ID)
        use_numpy: If True, save as numpy arrays. If False, save as PyTorch tensors
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Create a metadata dictionary to store shapes and types
    metadata = {}
    file_ext = ".pt.gz"
    layer_name = 'blocks.5.pool'
    safe_name = layer_name.replace('.', '_').replace('/', '_')
    base_dir = os.path.join(save_dir, safe_name)
    os.makedirs(base_dir, exist_ok=True)
    prefix_with_counter = f"{prefix}_tr_{counter}"
    safe_name = f"{prefix_with_counter}_{safe_name}"
    
    with gzip.open(os.path.join(base_dir, safe_name + file_ext), 'wb') as f:
        pickle.dump(embedding.cpu(), f)
    metadata[layer_name] = {
                'type': 'np',
                'shape': list(embedding.shape) if hasattr(embedding, 'shape') else None
    }
    utils.save_embedding_metadata(prefix_with_counter, metadata)
        