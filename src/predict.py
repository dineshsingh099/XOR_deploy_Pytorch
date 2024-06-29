import pandas as pd
import numpy as np
import torch
import train_pipeline as tp

X_test = pd.DataFrame(data={"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1]})
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

with torch.no_grad():
    predictions = tp.functional_nn(X_test_tensor)
    predictions = predictions.numpy() 

print("Predictions:")
print(predictions)

binary_predictions = (predictions > 0.5).astype(int)
print("Binary Predictions:")
print(binary_predictions)