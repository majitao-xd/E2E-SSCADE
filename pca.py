import numpy as np


def PCA(data):
    data = data.astype(np.float64)
    np.set_printoptions(suppress=True)
    num_bands = 3
    size = data.shape
    data_2d = (np.reshape(data, [size[0]*size[1], size[2]]))
    mu = np.mean(data_2d, axis=0)
    data_2d = data_2d - mu
    V = np.cov(data_2d.T)
    eigvec = pcacov(V)
    # latent, eigvec = np.linalg.eig(V)
    PC = np.zeros([size[0]*size[1], num_bands])
    for i in range(num_bands):
        PC[:, i] = np.matmul(data_2d, eigvec[:, i])
        PC[:, i] = ((PC[:, i] - np.mean(PC[:, i])) / np.std(PC[:, i]) + 3)*1000/6
        PC[:, i] = np.clip(PC[:, i], 0, 1000)
    PCs = (np.reshape(PC, [size[0], size[1], num_bands]))
    # return np.around(PCs)
    return PCs


def pcacov(cov):
    _, _, coeff = np.linalg.svd(cov)
    coeff = coeff.T
    size = cov.shape
    maxind = np.argmax(np.abs(coeff), axis=0)
    coeff = np.reshape(coeff, [size[0]*size[1]], order='F')
    aind = np.linspace(0, (size[1]-1)*size[0], size[0])
    ind = (maxind+aind).astype(np.int16)
    colsign = np.sign(coeff[ind])
    colsign = np.reshape(np.repeat(colsign, size[0]), size, order='F')
    eig = np.reshape(coeff,size,order='F')*colsign
    return eig
