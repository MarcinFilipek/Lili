#!/usr/bin/env python3
"""
Causal Chain Learning with Distributed Representations

This demonstrates how neural networks with distributed representations
can learn and compose causal rules:
  Rule 1: X→Y ⇒ A
  Rule 2: A→B ⇒ C
  Inference: X→Y→B ⇒ C (chain composition)

The key insight: intermediate predictions become inputs for next rules.
"""

import numpy as np

def print_header(title, char="="):
    """Pretty print section header."""
    print("\n" + char*70)
    print(title)
    print(char*70)

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def main():
    # ===== SETUP =====
    print_header("CAUSAL CHAIN LEARNING WITH DISTRIBUTED REPRESENTATIONS")
    
    # State vectors (3D for visualization, using ORTHOGONAL vectors)
    # This prevents interference between rules
    vec_X = np.array([1.0, 0.0, 0.0])
    vec_Y = np.array([0.0, 1.0, 0.0])
    vec_A = np.array([0.0, 0.0, 1.0])  # ORTHOGONAL to X,Y
    vec_B = np.array([1.0, 0.0, 0.0])  # Reuse (different context)
    
    # For 3D we need creative solution - let's use 5D instead
    state_dim = 5
    vec_X = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    vec_Y = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    vec_A = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # Intermediate
    vec_B = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    vec_C = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # Goal
    
    # Weight tensor: W[i,j,k]
    # Interpretation: "neuron i AND neuron j activate neuron k"
    # Weight tensor: W[i,j,k]
    # Interpretation: "neuron i AND neuron j activate neuron k"
    W = np.zeros((state_dim, state_dim, state_dim))
    lr = 0.2  # learning rate (increased for better predictions)
    
    print(f"\nState vectors:")
    print(f"  vec_X (start):    {vec_X}")
    print(f"  vec_Y (action):   {vec_Y}")
    print(f"  vec_A (intermediate): {vec_A}")
    print(f"  vec_B (action):   {vec_B}")
    print(f"  vec_C (goal):     {vec_C}")
    
    # ===== PHASE 1: Learn X→Y ⇒ A =====
    print_header("PHASE 1: Learning Rule 1: X→Y ⇒ A")
    
    print(f"\nObservation: When state is X and action is Y, outcome is A")
    print(f"Training this association 5 times (Hebbian learning)...")
    
    for epoch in range(5):
        # Hebbian update: ΔW[i,j,k] = η * X[i] * Y[j] * A[k]
        # This is the outer product: X ⊗ Y ⊗ A
        dW = lr * np.einsum('i,j,k->ijk', vec_X, vec_Y, vec_A)
        W += dW
        
        if epoch == 0:
            print(f"\nEpoch 1 - Example weight changes:")
            print(f"  ΔW[0,1,0] = {lr} * {vec_X[0]} * {vec_Y[1]} * {vec_A[0]} = {dW[0,1,0]:.4f}")
            print(f"  ΔW[0,1,1] = {lr} * {vec_X[0]} * {vec_Y[1]} * {vec_A[1]} = {dW[0,1,1]:.4f}")
            print(f"  ΔW[0,1,2] = {lr} * {vec_X[0]} * {vec_Y[1]} * {vec_A[2]} = {dW[0,1,2]:.4f}")
            print(f"\n  (These weights encode: 'when X[0] and Y[1] are active, activate A[k]')")
    
    print(f"\nAfter 5 training epochs, strongest weights:")
    print(f"  W[0,1,0] = {W[0,1,0]:.4f}  ← encodes X[0] ∧ Y[1] → A[0]")
    print(f"  W[0,1,1] = {W[0,1,1]:.4f}  ← encodes X[0] ∧ Y[1] → A[1]")
    print(f"  W[0,1,2] = {W[0,1,2]:.4f}  ← encodes X[0] ∧ Y[1] → A[2]")
    
    # Test prediction
    predicted_A = np.einsum('ijk,i,j->k', W, vec_X, vec_Y)
    print(f"\n✓ Prediction test: X→Y ⇒ ?")
    print(f"  Predicted: {predicted_A}")
    print(f"  Actual A:  {vec_A}")
    print(f"  L2 Error: {np.linalg.norm(predicted_A - vec_A):.4f}")
    print(f"  Cosine similarity: {cosine_similarity(predicted_A, vec_A):.4f}")
    
    # ===== PHASE 2: Learn A→B ⇒ C =====
    print_header("PHASE 2: Learning Rule 2: A→B ⇒ C")
    
    print(f"\nObservation: When state is A and action is B, outcome is C")
    print(f"Training this association 5 times...")
    
    for epoch in range(5):
        # Hebbian update: ΔW[i,j,k] = η * A[i] * B[j] * C[k]
        dW = lr * np.einsum('i,j,k->ijk', vec_A, vec_B, vec_C)
        W += dW
        
        if epoch == 0:
            print(f"\nEpoch 1 - Example weight changes:")
            # Find strongest weights
            max_idx = np.unravel_index(np.argmax(dW), dW.shape)
            i, j, k = max_idx
            print(f"  ΔW[{i},{j},{k}] = {lr} * {vec_A[i]:.1f} * {vec_B[j]:.1f} * {vec_C[k]:.1f} = {dW[i,j,k]:.4f} (strongest)")
    
    # Show some key weights after training rule 2
    print(f"\nKey weights after training both rules:")
    print(f"  W[0,1,0] = {W[0,1,0]:.4f}  ← from rule 1 (X→Y⇒A)")
    print(f"  W[0,1,2] = {W[0,1,2]:.4f}  ← from rule 2 (A→B⇒C)")
    print(f"  W[1,1,2] = {W[1,1,2]:.4f}  ← from rule 2 (A→B⇒C)")
    
    # Test prediction
    predicted_C = np.einsum('ijk,i,j->k', W, vec_A, vec_B)
    print(f"\n✓ Prediction test: A→B ⇒ ?")
    print(f"  Predicted: {predicted_C}")
    print(f"  Actual C:  {vec_C}")
    print(f"  L2 Error: {np.linalg.norm(predicted_C - vec_C):.4f}")
    print(f"  Cosine similarity: {cosine_similarity(predicted_C, vec_C):.4f}")
    
    # ===== PHASE 3: CHAIN INFERENCE =====
    print_header("PHASE 3: CAUSAL CHAIN INFERENCE - X→Y→B ⇒ ?", "=")
    
    print("\n⚠️  IMPORTANT: Agent has NEVER seen the sequence X→Y→B!")
    print("\nWhat agent learned:")
    print("  Rule 1: X→Y ⇒ A")
    print("  Rule 2: A→B ⇒ C")
    print("\nQuestion: Can agent infer that X→Y→B ⇒ C?")
    print("(This requires composing two learned rules)")
    
    # Step 1: Predict intermediate from X→Y
    print_header("Step 1: Predict intermediate state from X→Y", "-")
    intermediate = np.einsum('ijk,i,j->k', W, vec_X, vec_Y)
    print(f"Input:  vec_X = {vec_X}, vec_Y = {vec_Y}")
    print(f"Output: intermediate = {intermediate}")
    print(f"\nCompare to actual vec_A = {vec_A}")
    print(f"Cosine similarity: {cosine_similarity(intermediate, vec_A):.4f}")
    print(f"→ The intermediate state is similar to A (as expected from rule 1)")
    
    # Step 2: Use intermediate with B to predict final
    print_header("Step 2: Use intermediate state with B to predict final outcome", "-")
    print(f"Input:  intermediate = {intermediate}, vec_B = {vec_B}")
    final_outcome = np.einsum('ijk,i,j->k', W, intermediate, vec_B)
    print(f"Output: final_outcome = {final_outcome}")
    print(f"\nCompare to actual vec_C = {vec_C}")
    print(f"Cosine similarity: {cosine_similarity(final_outcome, vec_C):.4f}")
    
    # Normalized comparison
    final_norm = final_outcome / (np.linalg.norm(final_outcome) + 1e-10)
    vec_C_norm = vec_C / np.linalg.norm(vec_C)
    
    # ===== RESULTS =====
    print_header("FINAL RESULTS", "=")
    
    print(f"\nQuery: X→Y→B ⇒ ?")
    print(f"\nPredicted (normalized): {final_norm}")
    print(f"Actual C (normalized):  {vec_C_norm}")
    print(f"\nMetrics:")
    print(f"  L2 Error (raw):        {np.linalg.norm(final_outcome - vec_C):.4f}")
    print(f"  Cosine similarity:     {cosine_similarity(final_norm, vec_C_norm):.4f}")
    print(f"  Element-wise diff:     {np.abs(final_norm - vec_C_norm)}")
    
    similarity = cosine_similarity(final_norm, vec_C_norm)
    if similarity > 0.9:
        print("\n" + "="*70)
        print("✓✓✓ SUCCESS! Agent correctly inferred causal chain! ✓✓✓")
        print("="*70)
        print("\nThe agent:")
        print("  1. Never saw sequence X→Y→B during training")
        print("  2. Learned X→Y⇒A and A→B⇒C separately")
        print("  3. Successfully composed these rules to infer X→Y→B⇒C")
        print("\nThis demonstrates compositional causal reasoning!")
    else:
        print("\n✗ Inference accuracy could be better")
        print(f"  (cosine similarity {similarity:.4f} < 0.9)")
    
    # ===== WEIGHT ANALYSIS =====
    print_header("WEIGHT ANALYSIS - Top 10 Strongest Connections")
    
    # Find strongest weights
    weights_flat = []
    for i in range(state_dim):
        for j in range(state_dim):
            for k in range(state_dim):
                if W[i,j,k] > 0.01:  # Only significant weights
                    weights_flat.append((i, j, k, W[i,j,k]))
    
    weights_flat.sort(key=lambda x: x[3], reverse=True)
    
    print("\nStrongest learned associations:")
    for idx, (i, j, k, weight) in enumerate(weights_flat[:10]):
        print(f"  {idx+1}. W[{i},{j},{k}] = {weight:.4f}")
    
    print("\n" + "="*70)
    print("END OF DEMONSTRATION")
    print("="*70)

if __name__ == "__main__":
    main()
