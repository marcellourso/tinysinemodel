import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Settings
nsamples = 1000     # Number of samples to use as a dataset
val_ratio = 0.2     # Percentage of samples that should be held for validation set

# Generate some random samples
np.random.seed(1234)
x_values = np.random.uniform(low=0, high=(2 * math.pi), size=nsamples)

# Create a noisy sinewave with these values
y_values = np.sin(x_values) + (0.1 * np.random.randn(x_values.shape[0]))

# Split the dataset into training and validation sets
val_split = int(val_ratio * nsamples)
x_val, x_train = np.split(x_values, [val_split])
y_val, y_train = np.split(y_values, [val_split])

# Check that our splits add up correctly
assert(x_train.size + x_val.size) == nsamples

# Save the training data to sinetrain.csv
train_data = pd.DataFrame({
    'x': x_train,
    'y': y_train
})
train_data.to_csv('sinetrain.csv', index=False)

# Save the validation data to sinevalidate.csv
validation_data = pd.DataFrame({
    'x': x_val,
    'y': y_val
})
validation_data.to_csv('sinevalidate.csv', index=False)

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_val, y_val, 'y.', label="Validate")
plt.legend()
plt.show()
