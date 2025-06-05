#!/usr/bin/env python3
from typing import Iterable, Union
import math
Number = Union[int, float]


class Value:
    # -------------------------------------------------------------- T1
    def __init__(self, data: Number, _children: Iterable["Value"] = (), _op: str = ""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self.prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other: Union["Value", Number]):
        """self * other."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, other: Union["Value", Number]):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
    
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            self.grad += 0 
            if self.data > 0:
                other.grad += math.log(self.data) * out.data * out.grad
    
        out._backward = _backward
        return out
        
    def relu(self):
        """ReLU activation: max(0, x)."""
        out = Value(max(self.data, 0), (self,), "relu")
        def _backward():
            self.grad += max(0, out.grad)
        out._backward = _backward
        return out

    def backward(self):
        """Compute ``d(output)/d(node)`` for *every* ``node`` that influences
        this Value (call it *out*).

        Behaviour overview
        ------------------
        *The chain rule tells us we must process nodes **in reverse
        topological order** – children **before** parents – so that when
        we reach a node, the gradients flowing into it from all of its
        consumers have already been accumulated.*

        Implementation recipe
        ---------------------
        1. **Topological sort**
           Depth‑first search starting from ``self`` collects nodes in a
           list ``topo`` such that parents appear **before** children.
        2. **Seed the output**
           A node’s gradient with respect to itself is 1, so set
           ``self.grad = 1.0``.
        3. **Reverse sweep**
           Iterate ``for v in reversed(topo): v._backward()``.  Each
           stored ``_backward`` closure adds its *local* contribution to
           ``child.grad``.
        """
        # so we start from my node and go backwards calling backwards on all nodes from
        # root me, all the way to the leaves

        self.grad = 1
        topographical_ancestors = []
        seen = set()
        def recurse(node):
            topographical_ancestors.append(node)
            seen.add(node)
            if(node.prev):
                for node in node.prev:
                    if(node not in seen):
                        recurse(node)
        recurse(self)
        for node in topographical_ancestors:
            node._backward()
            
    def __rpow__(self, base: Union["Value", Number]):
        base = base if isinstance(base, Value) else Value(base)
        out = Value(base.data ** self.data, (base, self), '**')
    
        def _backward():
            base.grad += self.data * (base.data ** (self.data - 1)) * out.grad
            self.grad += math.log(base.data) * out.data * out.grad
    
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other: Union["Value", Number]):
        return self + (-other)

    def __rsub__(self, other: Union["Value", Number]):
        return other + (-self)

    def __truediv__(self, other: Union["Value", Number]):
        return self * other ** -1

    def __rtruediv__(self, other: Union["Value", Number]):
        return other * self ** -1

    def __repr__(self):
        return str(self.data)
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
    
        return out


## create the neuron, layer and MLP
import random
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlinearity):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if(self.nonlinearity == "relu"):
            return act.relu()
        return act

    def parameters(self):
        return self.w + [self.b]
    pass
    
class Fully_Connected_Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, nonlinearity):
        sz = [nin] + nouts
        self.layers = [Fully_Connected_Layer(sz[i], sz[i+1], nonlinearity=nonlinearity) for i in range(len(nouts)- 1)]
        self.layers.append(Fully_Connected_Layer(sz[-2], sz[-1], nonlinearity="none"))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
## bring in MNIST and setup some convenience functions
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Optionally normalize (pixel values between 0 and 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert to plain lists (if you don't want NumPy arrays)
x_train_list = x_train.tolist()
y_train_list = y_train.tolist()
print("done")

# let's work with a smaller set
# 28 x 28 examples
examples = 5000
x = x_train_list[:examples]
y = y_train_list[:examples]

# helper function to visualize an example
import matplotlib.pyplot as plt
import numpy as np

def show_mnist_example(image, pred_label=None, true_label=None):
    img = np.array(image)
    # Set up label text
    title = ""
    if true_label is not None:
        title += f"True: {true_label}  "
    if pred_label is not None:
        title += f"Pred: {pred_label}"

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=12)
    plt.show()

e = 2.71828

def loss_individual(model_outputs, target_label_index):
    max_output = max(model_outputs, key=lambda v: v.data if hasattr(v, 'data') else float(v))
    shifted_outputs = [val - max_output for val in model_outputs]

    exps = [Value(math.e) ** val for val in shifted_outputs]
    sum_exps = Value(0)
    for val in exps:
        sum_exps += val

    target_exp = exps[target_label_index]
    normalized_target = target_exp / sum_exps

    loss = -normalized_target.log()
    return loss

# test it on 10 examples
test_number = 20
# helper function to visualize an example
import matplotlib.pyplot as plt
import numpy as np

def pick_label(out):
    largest = 0
    index = 0
    for i in range(len(out)):
        if(out[i].data > largest):
            largest = out[i].data
            index = i
    return index


# now let's implement the convolutional layer and max /average pooling layer and see how those do
# if this becomes too slow, and I realize now that I may need to do a torch implementation because this autograd engine
# is very slow lol. I guess because I'm using python lists and non-vectorized code everywhere

class Convolutional_Unit(Module):
    def __init__(self, kernel_height, kernel_width, **kwargs):
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel = []
        for i in range(kernel_height):
            self.kernel.append([Value(random.uniform(-1,1)) for _ in range(kernel_width)])
        self.b = Value(0)
    
    def __call__(self, x):
        out_height = len(x) - self.kernel_height + 1
        out_width = len(x[0]) - self.kernel_width + 1
        # slide it over the input, return a 2d list of values
        out = []
        for i in range(out_height):
            row = []
            for j in range(out_width):
                # quadrouple nested for loop lol
                # there's definitely a lot ofrepeated work here, maybe I should make a map of values later and store them lol
                # or just vectorize the code lol
                acc = Value(0)
                for kernel_i in range(self.kernel_height):
                    for kernel_j in range(self.kernel_width):
                        # print(self.kernel[kernel_i][kernel_j])
                        # print(x[i + kernel_i][j + kernel_j])
                        acc += self.kernel[kernel_i][kernel_j] * x[i + kernel_i][j + kernel_j]
                acc += self.b
                row.append(acc)
            out.append(row)
        return out

    def parameters(self):
        params = []
        for row in self.kernel:
            for param in row:
                params.append(param)
        return params + [self.b]

class Convolutional_Feature_Map(Module):
    def __init__(self, num_units, kernel_height, kernel_width, **kwargs):
        self.convolutional_units = [Convolutional_Unit(kernel_height, kernel_width) for _ in range(num_units)]
        self.num_units = num_units

    def __call__(self, x):
        out = []

        is_list_of_inputs = isinstance(x, list) and isinstance(x[0][0], list)

        for i in range(self.num_units):
            inp = x[i] if is_list_of_inputs else x
            out.append(self.convolutional_units[i](inp))

        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for c in self.convolutional_units for p in c.parameters()]
    
class Average_Pooling_Unit(Module):
    def __init__(self, window_height, window_width, stride_height, stride_width):
        self.window_width = window_width
        self.window_height = window_height
        self.stride_height = stride_height
        self.stride_width = stride_width

    def __call__(self, x):
        # for simplicity we have to ensure that the input size % stride == 0
        if(len(x) % self.stride_height != 0 or len(x[0]) % self.stride_width != 0):
            print("need to make it divisible bruh")
            return
        out_height = len(x) // self.stride_height
        out_width = len(x[0]) // self.stride_width

        out = [[0 for _ in range(out_width)] for _ in range(out_height) ]
        
        
        for i in range(out_height):
            for j in range(out_width):
                start_i = i * self.stride_height
                start_j = j * self.stride_width
                acc = Value(0)
                for k in range(self.window_height):
                    for l in range(self.window_width):
                        acc += x[start_i + k][start_j + l]
                out[i][j] = acc / (self.window_width * self.window_height)
        return out
                
    def parameters(self):
        return []

class Average_Pooling_Feature_Map(Module):
    def __init__(self, num_units, window_height, window_width, stride_height, stride_width, **kwargs):
        self.pooling_units = [Average_Pooling_Unit(window_height, window_width, stride_height, stride_width) for _ in range(num_units)]
        self.num_units = num_units

    def __call__(self, x):
        out = []
        for i in range(self.num_units):
            out.append(self.pooling_units[i](x[i]))
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return []

class Flatten(Module):
    def __call__(self, x):
        # hard coding this to take in a 3d array and put it to 1d
        out = []
        for i in range(len(x)):
            for j in range(len(x[0])):
                for k in range(len(x[0][0])):
                    out.append(x[i][j][k])
        return out
            
        
    def parameters(self):
        return []
        

class LeNet(Module):
    def __init__(self):
      self.layers =[
          # start by average pooling it down to reduce params?
          Convolutional_Feature_Map(6, 3, 3), # output shape should be (6, 26, 26)
          Average_Pooling_Feature_Map(6, 2, 2, 2, 2), # output shape should be (6 x (13 x 13))
          Convolutional_Feature_Map(6, 2, 2), # output shape should be (6 x (12 x 12))
          Flatten(),
          Fully_Connected_Layer(864, 32,nonlinearity="relu"),
          Fully_Connected_Layer(32, 10,nonlinearity="none"),
      ]
      
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # print([p for layer in self.layers for p in layer.parameters()])
        return [p for layer in self.layers for p in layer.parameters()]


        
# we'll also need some sort of class that can wrap things and chain them together


print("starting")
lenet = LeNet()
lossi = []
learning_rate = 0.005
epochs = 25
batch_size = 8

print(len(lenet.parameters()))

for epoch in range(epochs):
    # Sample a random batch of data
    batch_indices = random.sample(range(len(x)), batch_size)
    batch_x = [x[i] for i in batch_indices]
    batch_y = [y[i] for i in batch_indices]

    average_loss_cum = Value(0)

    for xi, yi in zip(batch_x, batch_y):
        lenet.zero_grad()
        out = lenet(xi)
        loss = loss_individual(out, yi)
        loss.backward()
        average_loss_cum += loss
        for param in lenet.parameters():
            param.data -= learning_rate * param.grad

    print(average_loss_cum / batch_size)
    lossi.append(average_loss_cum / batch_size)

print("done")

# test out how our new convolutional neural net learns
import matplotlib.pyplot as plt

lossi = [v.data for v in lossi]

plt.plot(lossi)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()


# get correct percentage on all examples
correct = 0

for i in range(25):
    forward_label = lenet(x[i])
    pred = pick_label(forward_label)
    true = y[i]

    if(i % 5 == 0):
        print("calculating...")
    
    if pred == true:
        correct += 1

accuracy = (correct / 25) * 100
print(f"Accuracy: {accuracy:.2f}%")
