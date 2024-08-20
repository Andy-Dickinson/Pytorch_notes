import torch
import numpy as np

"""
*******************
Different ways to initialise tensors (similar to matrices)
*******************
"""
# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data: {type(x_data)}\n {x_data} \n")

# From Numpy array
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other
np_array = np.array(data)
t = torch.from_numpy(np_array)
print(f"np_array: \n {np_array} \n")
print(f"tensor from np_array: \n {t} \n")

# Note arrays and tensors share memory
t.add_(1)
print(f"change to tensor changes np_array: \n {np_array} \n")

# NumPy arrays can also be created from tensors
t2 = torch.ones(5)
print(f"tensor: \n {t2} \n")
np_array2 = t2.numpy()
print(f"np_array from tensor: \n {np_array2} \n")

# Note arrays and tensors share memory
np.add(np_array2, 1, out=np_array2)
print(f"change to np_array changes tensor: \n {t2} \n")


# From another tensor
# -- new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

print(f"x_data dtype: {x_data.dtype}")
x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n dtype: {x_rand.dtype} \n")

# With random or constant values
# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

"""
*******************
Attributes of a Tensor
*******************
"""
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

"""
*******************
Operations on a Tensor
*******************
"""
# Moving to GPU which is typically higher speeds than CPU
# By default, tensors created on CPU, need to move to GPU if available using '.to' method
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# Slicing
print(f"Tensor:\n {tensor}")
print(f"first row tensor[0]: {tensor[0]}")
print(f"first column tensor[:,0]: {tensor[:, 0]}")  # careful how use, see below
print(f"last column tensor[...,-1]: {tensor[..., -1]}")
tensor[:, 1] = 0
print(f"\nafter changing 2nd column: \n{tensor}\n")

tensor_3d = torch.tensor([[[1, 2, 3],
                           [4, 5, 6]],

                          [[7, 8, 9],
                           [10, 11, 12]]])
print(f"3d tensor: \n{tensor_3d}")
print(
    f"Slice last ele of last dimension across all dimensions [...,-1]: \n{tensor_3d[..., -1]}")  # tensor([[ 3,  6], [ 9, 12]])
print(f"Slice last ele of 2nd dimension for all rows [:,-1]: \n{tensor_3d[:, -1]}")
print(f"Note [:,0] does not give 1st col in 3d tensor: \n{tensor_3d[:, 0]}")

# Joining
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]], dtype=torch.float32)
print(f"Tensor: \n {tensor} \n {tensor.shape} \n")

# torch.cat concatenates a sequence of tensors along a given dimension
joined_0 = torch.cat([tensor, tensor, tensor])  # dim=0
print(f"Joined along dim=0: \n {joined_0} \n {joined_0.shape}\n")
joined_1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Joined along dim=1: \n {joined_1} \n {joined_1.shape}\n\n")

# torch.stack concatenates a sequence of tensors along a new dimension
print(f"Tensor before stacking: \n {tensor} \n {tensor.shape} \n")
stack_0 = torch.stack([tensor, tensor, tensor])  # dim=0
print(f"Stacked along dim=0: \n {stack_0} \n {stack_0.shape}\n")
stack_1 = torch.stack([tensor, tensor, tensor], dim=1)
print(f"Stacked along dim=1: \n {stack_1} \n {stack_1.shape}\n\n")

# Matrix multiplication
# @ is matrix multiplication operator
print(f"Tensor: \n {tensor} \n")
print(f"Tensor transposed: \n {tensor.T} \n ")
y1 = tensor @ tensor.T  # matrix multiplication between tensor and its transpose
y2 = tensor.matmul(tensor.T)  # equivalent to previous operation

# another way to perform matrix multiplication
y3 = torch.rand_like(y1)  # initialises y3 to same shape and data type as y1, but with random values
torch.matmul(tensor, tensor.T, out=y3)  # performs matrix multiplication (same as above), then stores result in y3
print(f"y3: \n {y3} \n ")

# Element-wise product
z1 = tensor * tensor  # element-wise multiplication
z2 = tensor.mul(tensor)  # equivalent to previous operation

# another way to perform element-wise multiplication
z3 = torch.rand_like(tensor)  # initialises z3 to same shape and data type as 'tensor' but with random values
torch.mul(tensor, tensor, out=z3)  # performs element-wise multiplication (same as above), then stores result in z3
print(f"z3: \n {z3} \n")

# Convert single-element tensors to Python numerical value using .item()
agg = tensor.sum()  # example of aggregating all values of a tensor into one value
print(f"Aggregated tensor: {agg} {type(agg)}")
agg_item = agg.item()  # converts to float
print(f"Converted back to Python numerical value: {agg_item} {type(agg_item)}")

# In place operations denoted by _ suffix
# operations that store result in operand are denoted by a _ suffix
x = torch.tensor([[1, 2, 3],
                 [4, 5, 6]])
y = torch.tensor([[7, 8, 9],
                 [10, 11, 12]])
print(f"x before copy: \n {x} \n")
print(f"y: \n {y} \n")
x.copy_(y)  # copies tensor y to tensor x - shapes must match
print(f"x after copy: \n {x} \n")
x.t_()  # transposes x and stores back to x
print(f"x after transpose: \n {x} \n")
x.add_(5)  # adds 5 to every element in x and stores back to x
print(f"x after adding 5: \n {x} \n")
