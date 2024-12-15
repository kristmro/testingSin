from famel import FAMLE
from tasks import SinusoidRegressionTaskDistribution
import tensorflow as tf

# Define task distribution
task_dist = SinusoidRegressionTaskDistribution()  # Custom implementation based on your data

# Define model architecture
base_model = tf.keras.model.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Initialize FAMLE
famle = FAMLE(
    base_model,
    taskDistribution=task_dist,
    outerLearningRate=0.001,
    innerLearningRate=0.01,
    nTasks=10,
    embeddingDim=5,
    nInnerSteps=5
)

# Train
famle.trainBatch(nSamples=20, nTasks=10, nBatch=1000)
