# Self-Pruning Neural Network
Implementation of a self-pruning neural network using learnable gates, L1 sparsity regularization, and PyTorch to dynamically remove unimportant weights during training.


## Approach
- Custom PrunableLinear layer with learnable gates  
- Sigmoid function used to control weight importance  
- L1 regularization applied on gate values to induce sparsity



## Loss Function
Total Loss = CrossEntropy Loss + λ × L1(gates)




## Results
| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.0001 | 43.11       | 1.49         |
| 0.001  | 40.29       | 1.72         |
| 0.01   | 37.78       | 1.72         |




## Key Insight
Increasing λ reduces model accuracy while slightly improving sparsity, showing a trade-off between performance and pruning.


## Gate Distribution (Different Lambda Values)
### λ = 0.0001
![Lambda 0.0001](graph_lambda_0.0001.png)

### λ = 0.001
![Lambda 0.001](graph_lambda_0.001.png)

### λ = 0.01
![Lambda 0.01](graph_lambda_0.01.png)
