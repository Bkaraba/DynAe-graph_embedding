from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DynAe(tf.keras.Model):
    def __init__(self, node_count, embedding_size, encoding_layer_sizes):
        super(DynAe, self).__init__()
        self.node_count = node_count
        self.embedding_size = embedding_size

        # Encoder layers
        self.encoding_layers = []
        for i, size in enumerate(encoding_layer_sizes):
            self.encoding_layers.append(tf.keras.layers.Dense(size, activation='relu'))

        # Embedding layer
        self.embedding_layer = tf.keras.layers.Dense(embedding_size, activation='relu')

        # Decoder layers
        self.decoding_layers = []
        for i, size in reversed(list(enumerate(encoding_layer_sizes))):
            self.decoding_layers.append(tf.keras.layers.Dense(size, activation='relu'))

        # Output layer
        self.output_layer = tf.keras.layers.Dense(node_count, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.encoding_layers:
            x = layer(x)
        x = self.embedding_layer(x)
        for layer in self.decoding_layers:
            x = layer(x)
        return self.output_layer(x)


# Download and process data at './dataset/ogbg-molhiv/'
dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv')

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)

# Define the number of nodes in the graph and the desired embedding size
node_count = 41127 * 25.5
embedding_size = 128

# Define the model
model = DynAe(node_count=node_count, embedding_size=embedding_size, encoding_layer_sizes=[32, 16])

# Compile the model with an optimizer and loss function
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy')

# Generate the data
X = np.random.randn(10000, 100).astype(np.float32)
y = np.random.randint(10, size=(10000, 1))
y = tf.keras.utils.to_categorical(y, num_classes=1048738)

# Train the model
epochs = 1
batch_size = 32

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        model.fit(X_batch, y_batch)

embeddings = model.layers[-4].weights[0].numpy()
print(embeddings)

# Reduce the embeddings to 2 dimensions using t-SNE
tsne = TSNE(n_components=2)
reduced_embeddings = tsne.fit_transform(embeddings)

#Plot the reduced embeddings in a scatter plot
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
plt.show()
