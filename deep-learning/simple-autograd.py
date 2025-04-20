import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import math


    class Variable:
        def __init__(self, val: float):
            self.val = val
            self.grad = 0
            self.backward_fn = None

        def backward(self, grad):
            if self.backward_fn:
                self.backward_fn(grad)
            else:
                self.grad += grad

        def __add__(self, y):
            o = Variable(self.val + y.val)
            o.backward_fn = AddBackwardFn(self, y, o.val)
            return o

        def __sub__(self, y):
            o = Variable(self.val - y.val)
            o.backward_fn = SubBackwardFn(self, y, o.val)
            return o

        def __mul__(self, y):
            o = Variable(self.val * y.val)
            o.backward_fn = MulBackwardFn(self, y, o.val)
            return o

        def __truediv__(self, y):
            o = Variable(self.val / y.val)
            o.backward_fn = DivBackwardFn(self, y, o.val)
            return o


    def exp(x: Variable):
        o = Variable(math.exp(x.val))
        o.backward_fn = ExpBackwardFn(x, o.val)
        return o


    def log(x: Variable):
        o = Variable(math.log(x.val))
        o.backward_fn = LogBackwardFn(x, o.val)
        return o


    def pow(x: Variable, e: float):
        o = Variable(math.pow(x.val, e))
        o.backward_fn = PowBackwardFn(x, o.val, e)
        return o


    class AddBackwardFn:
        def __init__(self, x: Variable, y: Variable, o: float) -> None:
            self.x = x
            self.y = y
            self.o = o

        def __call__(self, grad):
            self.x.backward(grad)
            self.y.backward(grad)


    class SubBackwardFn:
        def __init__(self, x: Variable, y: Variable, o: float) -> None:
            self.x = x
            self.y = y
            self.o = o

        def __call__(self, grad):
            self.x.backward(grad)
            self.y.backward(-grad)


    class MulBackwardFn:
        def __init__(self, x: Variable, y: Variable, o: float) -> None:
            self.x = x
            self.y = y
            self.o = o

        def __call__(self, grad):
            self.x.backward(self.y.val * grad)
            self.y.backward(self.x.val * grad)


    class DivBackwardFn:
        def __init__(self, x: Variable, y: Variable, o: float) -> None:
            self.x = x
            self.y = y
            self.o = o

        def __call__(self, grad):
            self.x.backward(grad/self.y.val)
            self.y.backward(- self.x.val * grad / self.y.val ** 2)


    class ExpBackwardFn:
        def __init__(self, x: Variable, o: float) -> None:
            self.x = x
            self.o = o

        def __call__(self, grad):
            self.x.backward(grad * self.o)


    class LogBackwardFn:
        def __init__(self, x: Variable, o: float) -> None:
            self.x = x
            self.o = o

        def __call__(self, grad):
            self.x.backward(grad / self.x.val)


    class PowBackwardFn:
        def __init__(self, x: Variable, o: float, e: float) -> None:
            self.x = x
            self.o = o
            self.e = e

        def __call__(self, grad):
            self.x.backward(grad * self.e * self.o / self.x.val)
    return (
        AddBackwardFn,
        DivBackwardFn,
        ExpBackwardFn,
        LogBackwardFn,
        MulBackwardFn,
        PowBackwardFn,
        SubBackwardFn,
        Variable,
        exp,
        log,
        math,
        pow,
    )


@app.cell
def _(Variable, pow):
    x = Variable(4.)
    y = Variable(3.)
    z = Variable(6.)


    w = pow(x * y, 2) + z / x

    w.backward(1.)

    print(x.grad)
    print(y.grad)
    print(z.grad)
    return w, x, y, z


if __name__ == "__main__":
    app.run()
