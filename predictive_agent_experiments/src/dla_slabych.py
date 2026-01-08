import numpy as np

# Vectors
vec_A = np.array([0.7, 0.2, 0.0])
vec_B = np.array([0.1, 0.2, 0.9])
vec_C = np.array([0.0, 0.7, 0.1])

# Initialize weights
W = np.zeros((3, 3, 3))
lr = 0.1

print("=== Training: A→B ⇒ C (5 times) ===\n")

for epoch in range(5):
    # Hebbian update
    dW = lr * np.einsum("i,j,k->ijk", vec_A, vec_B, vec_C)
    W += dW

    print(f"Epoch {epoch+1}:")
    print(f"  Strongest weight W[0,2,1] = {W[0,2,1]:.4f}")
    print(f"  Average weight = {np.mean(np.abs(W)):.4f}")

print("\n=== Prediction ===\n")

# Predict outcome
predicted = np.einsum("ijk,i,j->k", W, vec_A, vec_B)

print(f"Input state (A):  {vec_A}")
print(f"Input action (B): {vec_B}")
print(f"Predicted outcome: {predicted}")
print(f"Actual outcome (C): {vec_C}")
print(f"\nError: {np.linalg.norm(predicted - vec_C):.4f}")

# Normalized prediction
predicted_norm = predicted / (np.linalg.norm(predicted) + 1e-10)
vec_C_norm = vec_C / np.linalg.norm(vec_C)
print(f"\nNormalized predicted: {predicted_norm}")
print(f"Normalized actual:    {vec_C_norm}")
print(f"Cosine similarity: {np.dot(predicted_norm, vec_C_norm):.4f}")
