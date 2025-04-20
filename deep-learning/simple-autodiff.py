import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    return np, plt, random


@app.cell
def _(np):
    # Backward functions
    class AddBackward():
        def __init__(self,x,y,o):
            self.x = x
            self.y = y
            self.o = o
        def __call__(self,l):
            self.x.backward(l)
            self.y.backward(l)

    class SubBackward():
        def __init__(self,x,y,o):
            self.x = x
            self.y = y
            self.o = o
        def __call__(self,l):
            self.x.backward(l)
            self.y.backward(-l)

    class MulBackward():
        def __init__(self,x,y,o):
            self.x = x
            self.y = y
            self.o = o
        def __call__(self,l):
            self.x.backward(l * self.y.val)
            self.y.backward(l * self.x.val)

    class DivBackward():
        def __init__(self,x,y,o):
            self.x = x
            self.y = y
            self.o = o
        def __call__(self,l):
            self.x.backward(l / self.y.val)
            self.y.backward(- l * self.x.val / self.y.val ** 2)

    class PowBackward():
        def __init__(self,x,y,o):
            self.x = x
            self.y = y
            self.o = o
        def __call__(self,l):
            self.x.backward(l * self.y * self.o.val /self.x.val)

    class SigmoidBackward():
        def __init__(self,x,o):
            self.x = x
            self.o = o
        def __call__(self,l):
            self.x.backward(self.o.val*(1 - self.o.val) * l)

    class TanhBackward():
        def __init__(self,x,o):
            self.x = x
            self.o = o
        def __call__(self,l):
            self.x.backward(l * (1 - self.o.val ** 2) / 2)

    # Main data type: Variable
    class Variable():
        def __init__(self, x):
            self.val = x
            self.grad = 0
            self.backward_fn = None
        def zero_grad(self):
            self.grad = 0
        def backward(self,l):
            if self.backward_fn is not None:
                self.backward_fn(l)
            else:
                self.grad += l
        def __add__(self,x):
            o = Variable(self.val + x.val)
            o.backward_fn = AddBackward(self,x,o)
            return o
        def __sub__(self,x):
            o = Variable(self.val - x.val)
            o.backward_fn = SubBackward(self,x,o)
            return o
        def __mul__(self,x):
            o = Variable(self.val * x.val)
            o.backward_fn = MulBackward(self,x,o)
            return o
        def __truediv__(self,x):
            o = Variable(self.val / x.val)
            o.backward_fn = DivBackward(self,x,o)
            return o
        def __pow__(self,x):
            o = Variable(self.val ** x)
            o.backward_fn = PowBackward(self,x,o)
            return o
        def __str__(self):
            return str(self.val)

    # Activation Functions
    def sigmoid(x):
        o = Variable(1 / (1 + np.exp(-x.val)))
        o.backward_fn = SigmoidBackward(x,o)
        return o
    def tanh(x):
        o = Variable((1 - np.exp(-x.val)) / (1 + np.exp(-x.val)))
        o.backward_fn = TanhBackward(x,o)
        return o
    return (
        AddBackward,
        DivBackward,
        MulBackward,
        PowBackward,
        SigmoidBackward,
        SubBackward,
        TanhBackward,
        Variable,
        sigmoid,
        tanh,
    )


@app.cell
def _(np):
    # simple function helper to convert array of variables to numpy array
    def arr2nparray(arr):
        o = []
        for row in arr:
            tmp = []
            for elem in row:
                tmp.append(elem.val)
            o.append(tmp)
        return np.array(o)
    return (arr2nparray,)


@app.cell
def _(Variable, np, sigmoid):
    # Linear Layer
    class MyLinear():
        def __init__(self, in_features,out_features):
            self.W = []
            self.b = []
            self.in_features = in_features
            self.out_features = out_features
            for i in range(in_features):
                tmp = []
                for j in range(out_features):
                    tmp.append(Variable(np.random.randn()))
                self.W.append(tmp)
            for i in range(out_features):
                self.b.append(Variable(np.random.randn()))
        def forward(self, x):
            o = []
            for row in x:
                tmp = []
                for j in range(self.out_features):
                    s = Variable(0)
                    for i in range(self.in_features):
                        s = s + self.W[i][j] * row[i]
                    s += self.b[j]
                    tmp.append(s)
                o.append(tmp)
            return o
        def parameters(self):
            params = []
            for row in self.W:
                params += row
            params += self.b
            return params

    # Sigmoid Layer
    class MySigmoid():
        def __init__(self):
            pass
        def forward(self,x):
            o = []
            for row in x:
                tmp = []
                for e in row:
                    tmp.append(sigmoid(e))
                o.append(tmp)
            return o
        def parameters(self):
            return []

    # Sequential Layer
    class MySequential():
        def __init__(self,layers=[]):
            self.layers = layers
        def forward(self,x):
            o = x
            for l in self.layers:
                o = l.forward(o)
            return o
        def parameters(self):
            params = []
            for l in self.layers:
                params += l.parameters()
            return params
        def __call__(self,x):
            return self.forward(x)
    return MyLinear, MySequential, MySigmoid


@app.cell
def _(Variable):
    # Mean Square Error Loss
    class MyMSELoss():
        def __init__(self):
            pass
        def __call__(self,y, t):
            N = len(y)
            s = Variable(0)
            for row_y, row_t in zip(y,t):
                for yi,ti in zip(row_y,row_t):
                    s += (yi - ti) ** 2
            s /= Variable(N)
            return s
    return (MyMSELoss,)


@app.cell
def _():
    # Stochastic Gradient Descent Optimizer
    class MySGD():
        def __init__(self,parameters=[],lr=0.1):
            self.lr = lr
            self.parameters = parameters
        def step(self):
            for p in self.parameters:
                p.val -= self.lr * p.grad
        def zero_grad(self):
            for p in self.parameters:
                p.zero_grad()
    return (MySGD,)


@app.cell
def _(MyLinear, MyMSELoss, MySGD, MySequential, MySigmoid):
    # Creating our model
    model = MySequential(layers=[
        MyLinear(2,3),
        MySigmoid(),
        MyLinear(3,2)
    ])

    # Defining our loss function
    crit = MyMSELoss()

    # Defining our optimizer (SGD)
    optim = MySGD(parameters=model.parameters(),lr=0.1)
    return crit, model, optim


@app.cell
def _(Variable):
    # Dummy data
    x = [[Variable(1),Variable(1)]]
    y = [[Variable(3),Variable(2)]]
    return x, y


@app.cell
def _(crit, model, optim, x, y):
    for epoch in range(5):
        o = model(x)
        optim.zero_grad()
        l = crit(o,y)
        print("loss = ",l)
        l.backward(1)
        optim.step()
    return epoch, l, o


@app.cell
def _(arr2nparray, o):
    # Visualizing our output
    print(arr2nparray(o))
    return


if __name__ == "__main__":
    app.run()
