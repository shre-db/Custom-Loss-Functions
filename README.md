# Custom Loss Functions

![Loss Function Cover Image](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Plot_of_the_imaginary_error_function_Erfi%28z%29_in_the_complex_plane_from_-2-2i_to_2%2B2i_with_colors_created_with_Mathematica_13.1_function_ComplexPlot3D.svg/640px-Plot_of_the_imaginary_error_function_Erfi%28z%29_in_the_complex_plane_from_-2-2i_to_2%2B2i_with_colors_created_with_Mathematica_13.1_function_ComplexPlot3D.svg.png)

## Introduction
This repository contains a collection of custom loss functions designed for use in Artificial Neural Networks (ANNs). These loss functions are neither rigorously tested nor are regularly used in deep learning projects outside of research. The use of these custom loss functions in production is less common than standard loss functions. In many production systems, standard loss functions like Mean Squared Error (MSE) or Cross-Entropy Loss are preferred due to their simplicity and well understood properties. However, there are scenarios where custom loss functions are used in production, especially when the problem requirements demand it.

## L1 Loss
$$\mathcal{L}(\hat y, y) = n^{-1}\displaystyle\sum_{i=1}^{n}|\hat y_i - y_i|$$
where:<br>
$\hat y$ are predictions from the model.<br>
$y$ are ground truths or labels.<br>
$n$ is the number of observations.
- L1 loss, also known as the mean absolute error (MAE), measures the mean of absolute differences between predictions and ground truths.
- It's a good choice when you want to penalize large errors linearly, which makes it less sensitive to outliers compared to L2 loss (MSE).
- This loss can be useful for regression tasks where you want to emphasize the magnitude of errors rather than the squared errors.

## L2AveLoss
$$\mathcal{L}(\hat y, y) = n^{-1} \displaystyle\sum_{i=1}^{n}(\hat y_i - y_i)^2 + n^{-1}\Bigg |\displaystyle\sum_{i=1}^{n}\hat y_i \Bigg |$$
where:<br>
$\hat y$ are predictions from the model.<br>
$y$ are ground truths or labels.<br>
$n$ is the number of observations.
- L2 Average Loss (L2AveLoss) is a conjuctive loss function involving mean squared error (MSE) and the absolute average of predictions.
- The MSE part encourages accurate predictions and smooth gradients during training. The absolute average part encourages predictions to be centered around zero.
- This loss could be useful when you want a balance between accuracy and prediction centralization.

## CorrLoss
$$\mathcal{L}(\hat y, y) = -\frac{\displaystyle\sum(\hat y - \mu_{\hat y})(y - \mu_y)}{(n-1)\sigma_{\hat y}\sigma_{y}}$$
where:<br>
$\hat y$ are predictions from the model.<br>
$y$ are ground truths or labels.<br>
$n$ is the number of observations.<br>
$\mu_{\hat y}$ is the mean of predictions.<br>
$\mu_{y}$ is the mean of ground truths or labels.<br>
$\sigma_{\hat y}$ is the standard deviation of predictions.<br>
$\sigma_{y}$ is the standard deviation of ground truths or labels.<br>
- Correlation Loss (CorrLoss) measures the correlation between predictions and ground truths, which can be useful in scenarios where you want to maximize correlation.
- It encourages a specific statistical relationship between predictions and ground truths.
- It might be suitable for tasks where you have prior knowledge and ground truths.

## Implementation
All of the above loss functions are implemented in the notebook.ipynb file using PyTorch.