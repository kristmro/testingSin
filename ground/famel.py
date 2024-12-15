import tensorflow as tf
from model import Model

class FAMLE(Model):
    def __init__(self, *args, nInnerSteps=5, embeddingDim=5, nTasks=10, **kwargs):
        """
        Fast adaptation through meta learning embeddings (FAMLE) model.
        Args:
            embeddingDim: Dimensionality of the task embeddings.
            nTasks: Number of tasks.
            nInnerSteps: Number of inner gradient steps for adaptation.
        """
        super().__init__(*args, **kwargs)
        self.nInnerSteps = nInnerSteps
        self.embeddingDim = embeddingDim
        self.nTasks = nTasks

        # Initialize embeddings
        self.taskEmbeddings = tf.Variable(
            tf.random.normal([nTasks, embeddingDim]), trainable=True, name="task_embeddings"
        )

        self.interpolationRateInitial = self.outerLearningRate
        self.sgd = tf.keras.optimizers.SGD(self.innerLearningRate)
        self.modelCopy = tf.keras.models.clone_model(self.model)

    @tf.function
    def getTaskEmbedding(self, task_id):
        """Get the softmax-normalized embedding for a given task ID."""
        embeddings = tf.nn.softmax(self.taskEmbeddings, axis=0)
        return tf.expand_dims(embeddings[task_id], axis=0)

    @tf.function
    def interpolate(self, source, target):
        """Linearly interpolate between source and target."""
        return target + (source - target) * self.interpolationRate

    @tf.function
    def copyWeightsApply(self, source, target, fn=lambda s, t: s):
        """Assign weights from source to target."""
        for j in range(len(source.trainable_weights)):
            target.trainable_weights[j].assign(
                fn(source.trainable_weights[j], target.trainable_weights[j])
            )

    @tf.function
    def updateInterpolationRate(self):
        self.interpolationRate = self.interpolationRateInitial * (1 - self.nIteration / self.nIterations)
        self.nIteration += 1

    @tf.function
    def taskLoss(self, batch, task_id):
        """Compute task-specific loss."""
        y, x = batch

        # Extract task embedding
        task_embedding = self.getTaskEmbedding(task_id)

        # Add embedding to input (example architecture extension)
        x_with_embedding = tf.concat([x, tf.tile(task_embedding, [tf.shape(x)[0], 1])], axis=-1)

        self.copyWeightsApply(self.model, self.modelCopy)
        self.updateInterpolationRate()

        # Inner-loop training
        for _ in range(self.nInnerSteps):
            with tf.GradientTape() as taskTape:
                loss = self.lossfn(y, self.modelCopy(tf.reshape(x_with_embedding, (-1, 1))))

            self.sgd.minimize(
                loss, self.modelCopy.trainable_variables, tape=taskTape
            )

        self.copyWeightsApply(self.modelCopy, self.model, self.interpolate)
        return loss

    @tf.function
    def update(self, batch, task_ids):
        """Perform meta-update across tasks."""
        task_losses = tf.map_fn(
            lambda task: self.taskLoss(batch[task[0]], task[1]),
            elems=list(enumerate(task_ids)),
            fn_output_signature=tf.float32,
        )
        return tf.reduce_sum(task_losses)

    def trainBatch(self, nSamples, nTasks, nBatch):
        self.nIterations = nBatch
        self.nIteration = 0
        self.innerLearningRate = self.interpolationRateInitial

        # Sample task batches and perform meta-update
        batch = self.taskDistribution.sampleTaskBatches(nSamples, nTasks, nBatch)
        return float(tf.reduce_mean(
            tf.map_fn(
                lambda batch: self.update(batch[0], batch[1]),
                elems=batch,
                fn_output_signature=tf.float32
            )
        ))
