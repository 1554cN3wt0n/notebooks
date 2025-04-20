import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from sklearn.datasets import load_iris, load_digits
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    return load_digits, load_iris, np, plt, train_test_split


@app.cell
def _(np):
    # model parameters
    def init_parameters(input_size, hidden_neurons, num_classes):
        # layer 1 parameters (important to normalize the weights for convergence)
        W1 = np.random.rand(input_size,hidden_neurons) / (input_size * hidden_neurons)
        b1 = np.random.rand(hidden_neurons) / hidden_neurons

        # layer 2 parameters (important to normalize the weights for convergence)
        W2 = np.random.rand(hidden_neurons,num_classes) / (num_classes * hidden_neurons)
        b2 = np.random.rand(num_classes) / num_classes
        return W1, b1, W2, b2

    # activation function
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    # model forward
    def forward(inp, params):
        W1, b1, W2, b2 = params
        o0 = inp
        o1 = o0 @ W1 + b1
        o2 = sigmoid(o1)
        o3 = o2 @ W2 + b2
        return o0, o1, o2, o3

    # model backward
    def backward(dl_do3, params, o):
        W1, b1, W2, b2 = params
        o0, o1, o2, o3 = o

        do3_do2 = W2.T
        do3_dW2 = o2
        do3_db2 = 1

        do2_do1 = o2 * (1 - o2)

        do1_dW1 = o0
        do1_db1 = 1

        dl_do2 = dl_do3 @ do3_do2
        dl_dW2 = (dl_do3.T @ do3_dW2).T
        dl_db2 = np.sum(dl_do3 * do3_db2,axis=0)

        dl_do1 = dl_do2 * do2_do1

        dl_dW1 = (dl_do1.T @ do1_dW1).T
        dl_db1 = np.sum(dl_do1 * do1_db1,axis=0)

        return dl_dW1, dl_db1, dl_dW2, dl_db2
    return backward, forward, init_parameters, sigmoid


@app.cell
def _(np):
    # Loss function: Softmax + Cross Entropy
    def loss_fn(out, target):
        e_out = np.exp(out)
        softmax = e_out / np.sum(e_out,axis=1).reshape(-1,1)
        log_softmax = np.log(softmax)

        loss = -np.sum(log_softmax[np.arange(len(target)),target])
        mask = np.zeros_like(softmax)
        mask[np.arange(len(target)),target] = 1

        grad_loss = softmax - mask
        return loss, grad_loss
    return (loss_fn,)


@app.cell
def update_params():
    def update_params(params,dparams,lr=0.1):
        new_params = []
        for p,dp in zip(params,dparams):
            new_params.append(p - lr*dp)
        return tuple(new_params)
    return (update_params,)


@app.cell
def _(backward, forward, loss_fn, update_params):
    # Training
    def train(params, train_x, train_y, epochs=1000,lr=0.001):
        # History of the loss value in each epoch
        history = []

        for epoch in range(epochs):
            # Forwarding
            o = forward(train_x, params)

            # Calculating the loss and its gradient with respect to the last layer
            loss, dl_do3 = loss_fn(o[-1], train_y)
            history.append(loss)
            if epoch % 100 == 0:
                print(loss)

            # Backwarding the gradient of the loss
            dparams = backward(dl_do3, params, o)

            # Updating parameters
            params = update_params(params,dparams,lr=lr)
        return params, history
    return (train,)


@app.cell
def _(forward, init_parameters, load_iris, np, train, train_test_split):
    def iris_data():
        # Loading data
        raw_data = load_iris()
        x, y = raw_data.data, raw_data.target

        # Splitting data in train and test dataset
        train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.8)

        # input_size = 4
        # hidden_neurons = 5
        # num_classes = 3
        params = init_parameters(4,5,3)
        params, history = train(params, train_x, train_y, epochs=1000, lr=0.001)

        _,_,_,out = forward(test_x, params)
        print("predicted classes", np.argmax(out,axis=1))
        print("ground classes   ", test_y)
    return (iris_data,)


@app.cell
def _(iris_data):
    iris_data()
    return


@app.cell
def _(forward, init_parameters, load_digits, np, train, train_test_split):
    def digits_data():
        # Loading data
        raw_data = load_digits()
        x, y = raw_data.data, raw_data.target
    
        # Splitting data in train and test dataset
        train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.8)

        # input_size = 64
        # hidden_neurons = 128
        # num_classes = 10
        params = init_parameters(64,32,10)
        params, history = train(params, train_x, train_y, epochs=1000, lr=0.001)

        _,_,_,out = forward(test_x, params)
        print("predicted classes", np.argmax(out,axis=1))
        print("ground classes   ", test_y)
    return (digits_data,)


@app.cell
def _(digits_data):
    digits_data()
    return


if __name__ == "__main__":
    app.run()
