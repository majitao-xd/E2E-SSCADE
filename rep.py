import paddle
import numpy as np

class model(paddle.nn.Layer):
    def __init__(self,dim):
        super(model, self).__init__()
        #paddle.seed(4321)
        self.ly1 = paddle.nn.Conv2D(dim,3,(3,3),padding=1,padding_mode='circular')
        self.ly2 = paddle.nn.Conv2D(3,dim,(3,3),padding=1,padding_mode='circular')

    def forward(self,data):
        #print(data.shape)
        h1 = self.ly1(data)
        h1 = paddle.nn.functional.tanh(h1)
        return self.ly2(h1),h1

def REP(data):
    data = (data - data.min()) / (data.max() - data.min())
    size = data.shape
    ex = model(size[2])
    ex.train()
    optim = paddle.optimizer.Adam(1e-4, parameters=ex.parameters())
    data_in = paddle.to_tensor(data, 'float32')
    data_in = paddle.transpose(data_in, (2, 0, 1))
    data_in = paddle.expand(data_in, [1, size[2], size[0], size[1]])
    for i in range(20):
        r, h = ex(data_in)
        loss = paddle.nn.functional.mse_loss(r, data_in)
        loss.backward()
        optim.step()
        optim.clear_grad()
    h = h.numpy()
    h = np.squeeze(h)
    h = np.transpose(h, (1, 2, 0))
    return h
