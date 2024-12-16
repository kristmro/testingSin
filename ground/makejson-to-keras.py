import tensorflow as tf
import json
import numpy as np

# Paths to files
json_file_path = '/home/kristmro/workspace/MetaLearningCodes/TestingSin/models/fomaml-reproduce/model.json'
weights_file_path = '/home/kristmro/workspace/MetaLearningCodes/TestingSin/models/fomaml-reproduce/group1-shard1of1.bin'

# Load the JSON file
with open(json_file_path, 'r') as file:
    model_data = json.load(file)

# Extract the model topology and weights manifest
model_config = model_data['modelTopology']['model_config']
weights_manifest = model_data['weightsManifest'][0]['weights']

# Reconstruct the model architecture
model = tf.keras.Sequential.from_config(model_config['config'])

# Load weights from the binary file
with open(weights_file_path, 'rb') as weight_file:
    binary_weights = weight_file.read()

# Decode weights based on manifest
weights = []
offset = 0
for weight_info in weights_manifest:
    # Get the shape and size of each weight tensor
    shape = weight_info['shape']
    dtype = weight_info['dtype']
    dtype = np.float32 if dtype == 'float32' else np.float64
    size = np.prod(shape)

    # Extract and reshape the binary data for this weight
    weight_data = np.frombuffer(binary_weights, dtype=dtype, count=size, offset=offset)
    weights.append(weight_data.reshape(shape))
    offset += size * np.dtype(dtype).itemsize

# Set the weights into the model
model.set_weights(weights)

# Save the model in SavedModel format
output_path = '/home/kristmro/workspace/MetaLearningCodes/TestingSin/savedModels/fomaml-reproduce.keras'
model.save(output_path)
print(f"Model successfully saved to {output_path}")

