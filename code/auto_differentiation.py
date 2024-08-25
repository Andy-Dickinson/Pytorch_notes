import torch

"""
Autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) 
in a directed acyclic graph (DAG) consisting of Function objects. In this DAG, leaf nodes are the input tensors, 
roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the 
gradients using the chain rule.
In forward pass autograd:
    * runs requested operation to compute a resulting tensor
    * maintain the operation’s gradient function in the DAG
In backward pass (by calling .backward()) autograd:
    * computes the gradients by applying the chain rule to the operations stored in each .grad_fn
    * accumulates them in the respective tensor’s .grad attribute
    * propagates gradients all the way to the leaf tensors
    - note: each graph is recreated from scratch after each .backward() call unless retain_graph=True is passed in call

Below is a simple one-layer network with input x, parameters w, b and some loss function

Need to optimise params w and b. 
To compute gradients of loss function with respect to those variables, need to set 'requires_grad' property of those tensors.
Either set 'requires_grad' when creating tensor, or later with x.requires_grad_(True) method. 
Gradients will not be available for other graph nodes. 
Avoid in-place operations on tensors that require gradients as can cause errors during backpropagation.
"""
x = torch.ones(5)  # input tensor (vector of 5 ones)
y = torch.zeros(3)  # expected output (vector of 3 zeros)
w = torch.randn(5, 3, requires_grad=True)  # weight matrix, initialised randomly
b = torch.randn(3, requires_grad=True)  # bias vector, initialised randomly

z = torch.matmul(x, w)+b  # linear output (logits) of the network before applying any activation function
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # computes the loss (binary cross-entropy with logits)

# a function applied to tensors to construct computational graph is an object of 'Function' class,
# which knows how to compute the function in the forwards direction, and how to compute derivative during back propagation step
# reference to backward propagation function is stored in 'grad_fn'
print(f"Gradient function reference for z = {z.grad_fn}")
print(f"Gradient function reference for loss = {loss.grad_fn} \n")

# leaf tensors are tensors created by user that have 'requires_grad=True' and store their gradients in .grad attribute
print(f"w leaf tensor: {w.is_leaf}")
print(f"z leaf tensor: {z.is_leaf} \n")

# compute gradients: first call backward() then retrieve gradients from the .grad attributes
# can only perform gradient calculations using backward() once on a given graph by default, for performance reasons
# if we need to do several backward calls on the same graph, we need to pass 'retain_graph=True' to the backward() call
loss.backward()  # calculates gradient of the loss tensor wrt graph leaves (all tensors in graph with requires_grad=true)
print(f"Gradient with respect to w: \n {w.grad} \n")  # retrieve gradient of loss function with respect to w
print(f"Gradient with respect to b: \n {b.grad} \n")  # retrieve gradient of loss function with respect to b

# gradients are accumulated in '.grad' attribute
# beneficial for certain optimisation algs,
# but require manually zeroing between training iterations to prevent accumulations from multiple backward passes
# this is better done by zeroing all parameters on the optimizer object
w.grad.zero_()
b.grad.zero_()


# disable gradient tracking:
# all tensors with requires_grad=True are tracking their computational history and support gradient computation
# if only want to do forward computations (e.g. when we have trained the model and just want to apply it to some input data),
# can stop tracking to save memory and computation
z = torch.matmul(x, w)+b
print(f"tracking on: {z.requires_grad}")

with torch.no_grad():  # surround code to turn off gradient tracking
    z = torch.matmul(x, w)+b
print(f"tracking on: {z.requires_grad}")

z.requires_grad_(True)  # turns tracking back on
print(f"tracking on: {z.requires_grad}")

# alternatively detach a tensor from its computational graph
z_det = z.detach()  # creates a new tensor that shares same data but with tracking off
print(f"tracking on: {z_det.requires_grad}")

