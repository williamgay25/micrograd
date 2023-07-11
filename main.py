from micrograd.nn import MLP

# Initialize the NN
model = MLP(3, [4, 4, 1])

# Create some Data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Train the NN
for k in range(100):

    # forward
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    learning_rate = 0.25
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print(f"step {k} loss {loss.data} ypred {ypred[0].data}")