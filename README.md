# Neural Network from Scratch
*Learning and optimizing neural netowrk using the numpy only*

```

Input A_0 = X
    │
    ▼
┌──────────────────────────────────────────┐
│  For l = 1, 2, ..., L-1 (Hidden Layers)  │
│  • Z_l = W_l @ A_{l-1} + b_l             │
│  • A_l = ReLU(Z_l)                       │
│  • Store (A_{l-1}, Z_l, W_l, b_l) in Cache│
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  For l = L (Output Layer)                │
│  • Z_L = W_L @ A_{L-1} + b_L             │
│  • A_L = Softmax(Z_L)                    │
│  • Store (A_{L-1}, Z_L, W_L, b_L) in Cache│
└──────────────────┬───────────────────────┘
                   │
                   ▼
     Loss Evaluation: L(Y, A_L)


```
Loss L (from Y and A_L)
    │
    ▼
┌──────────────────────────────────────────┐
│  Step 1: Output Layer Gradient (l = L)   │
│  • dZ_L = A_L - Y                        │
│  • dW_L = (1/N) * dZ_L @ (A_{L-1})^T     │
│  • db_L = (1/N) * sum(dZ_L, axis=1)      │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Step 2: Loop for l = L-1 down to 1      │
│  • dA_l = (W_{l+1})^T @ dZ_{l+1}         │
│  • dZ_l = dA_l * relu_backward(Z_l)      │
│  • dW_l = (1/N) * dZ_l @ (A_{l-1})^T     │
│  • db_l = (1/N) * sum(dZ_l, axis=1)      │
└──────────────────────────────────────────┘