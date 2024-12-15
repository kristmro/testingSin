import numpy as np

weights_path = './models/maml-reproduce/group1-shard1of1.bin'

with open(weights_path, 'rb') as f:
    weights = np.frombuffer(f.read(), dtype=np.float32)

print("Loaded weights shape:", weights.shape)
