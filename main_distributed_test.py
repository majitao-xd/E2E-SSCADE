import time
import paddle
from paddle.io import Dataset
import paddle.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_auc_score
from calculate_falarm_rate import false_alarm_rate
#from pca import PCA
from rep import REP

class mydatasets(Dataset):
    def __init__(self, path):
        super(mydatasets, self).__init__()
        self.data_path = path
        self.data = loadmat(self.data_path)['data'].astype(np.float32)
        if 'fea' in loadmat(self.data_path).keys():
            self.fea = loadmat(self.data_path)['fea'].astype(np.float32)
        else:
            self.fea = REP(self.data)
            #print('gen rep')
            #plt.figure()
            #plt.imshow(self.fea[:,:,0])
            #plt.figure()
            #plt.imshow(self.fea[:,:,1])
            #plt.figure()
            #plt.imshow(self.fea[:,:,2])
            #plt.show()
        self.map = loadmat(self.data_path)['map']
        self.norm()
        self.size = self.map.shape
        self.data_dim = self.data.shape[-1]
        self.fea_dim = self.fea.shape[-1]
        self.num_sampales = self.size[0]*self.size[1]
        self.reshape()

    def norm(self):
        self.data = (self.data-np.min(self.data))/(np.max(self.data)-np.min(self.data))
        self.fea = (self.fea-np.min(self.fea))/(np.max(self.fea)-np.min(self.fea))

    def reshape(self):
        self.data = np.reshape(self.data, [self.num_sampales, self.data_dim])
        self.fea = np.reshape(self.fea, [self.num_sampales, self.fea_dim])

    def get_map(self):
        return self.map, self.map.shape

    def get_dim(self):
        return self.data_dim, self.fea_dim

    def __getitem__(self, item):
        return [self.data[item, :], self.fea[item, :]]

    def __len__(self):
        return self.num_sampales

class mymodel(paddle.nn.Layer):
    def __init__(self,num_clusters,hidden_nodes_1,hidden_nodes_2,est_hiddens,data_dim,fea_dim,w_cat,w_add,activate):
        super(mymodel,self).__init__()
        self.num_clusters = num_clusters
        self.hidden_nodes_1 = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.est_hiddens = est_hiddens
        self.data_dim = data_dim
        self.fea_dim = fea_dim
        self.w_cat = w_cat
        self.w_add = w_add
        self.seed = 2345
        #paddle.seed(self.seed)
        if activate == 'tanh':
            self.activate = paddle.nn.functional.tanh
        else:
            self.activate = paddle.nn.functional.relu6

        weight_init = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        # compression_net_1_pre
        self.layer_1_ae_1 = paddle.nn.Linear(self.data_dim,self.hidden_nodes_1[0],weight_attr=weight_init)
        self.layer_1_ae_2 = paddle.nn.Linear(self.hidden_nodes_1[0],self.hidden_nodes_1[1],weight_attr=weight_init)
        self.layer_1_ae_3 = paddle.nn.Linear(self.hidden_nodes_1[1],self.hidden_nodes_1[0],weight_attr=weight_init)
        self.layer_1_ae_4 = paddle.nn.Linear(self.hidden_nodes_1[0],self.data_dim,weight_attr=weight_init)

        # compression_net_2_pre
        self.layer_2_ae_1 = paddle.nn.Linear(self.fea_dim, self.hidden_nodes_2[0],weight_attr=weight_init)
        self.layer_2_ae_2 = paddle.nn.Linear(self.hidden_nodes_2[0], self.hidden_nodes_2[1],weight_attr=weight_init)
        self.layer_2_ae_3 = paddle.nn.Linear(self.hidden_nodes_2[1], self.hidden_nodes_2[0],weight_attr=weight_init)
        self.layer_2_ae_4 = paddle.nn.Linear(self.hidden_nodes_2[0], self.fea_dim,weight_attr=weight_init)

        # estimation_net_pre(self):
        self.layer_en_1 = paddle.nn.Linear(self.hidden_nodes_1[1]+2, self.est_hiddens,weight_attr=weight_init)
        self.layer_en_2 = paddle.nn.Linear(self.est_hiddens, self.num_clusters,weight_attr=weight_init)

    def compression_net_1_run(self,x):
        layer_1_ae_1 = self.layer_1_ae_1(x)
        layer_1_ae_1 = self.activate(layer_1_ae_1)
        layer_1_ae_2 = self.layer_1_ae_2(layer_1_ae_1)
        layer_1_ae_3 = self.layer_1_ae_3(layer_1_ae_2)
        layer_1_ae_3 = self.activate(layer_1_ae_3)
        layer_1_ae_4 = self.layer_1_ae_4(layer_1_ae_3)
        layer_1_ae_4 = self.activate(layer_1_ae_4)
        return layer_1_ae_2,layer_1_ae_4

    def compression_net_2_run(self,y):
        layer_2_ae_1 = self.layer_2_ae_1(y)
        layer_2_ae_1 = self.activate(layer_2_ae_1)
        layer_2_ae_2 = self.layer_2_ae_2(layer_2_ae_1)
        layer_2_ae_3 = self.layer_2_ae_3(layer_2_ae_2)
        layer_2_ae_3 = self.activate(layer_2_ae_3)
        layer_2_ae_4 = self.layer_2_ae_4(layer_2_ae_3)
        layer_2_ae_4 = self.activate(layer_2_ae_4)
        return layer_2_ae_2,layer_2_ae_4

    def extract_feature(self,x,z):
        #mse = F.mse_loss(x,z)
        norm_x = paddle.sqrt(paddle.sum(paddle.square(x),axis=1))
        #norm_z = paddle.sqrt(paddle.sum(paddle.square(z),axis=1))
        dist_x = paddle.sqrt(paddle.sum(paddle.square(x-z),axis=1))
        min_val = 1e-12
        loss_e = dist_x/(norm_x+min_val)
        loss_opd = self.OPD(x,z)
        loss_e = paddle.reshape(loss_e,[loss_e.shape[0],1])
        loss_opd = paddle.reshape(loss_opd, [loss_opd.shape[0], 1])
        return paddle.concat([loss_e,loss_opd], axis=1)

    def OPD(self,x,z):
        ri = paddle.transpose(x,[1,0])
        rj = paddle.transpose(z,[1,0])
        L = ri.shape[0]
        I = paddle.ones([L,1])
        ones = paddle.ones([L,1])
        diag1 = paddle.diag(paddle.matmul(x,ri))
        diag2 = paddle.diag(paddle.matmul(z,rj))
        diag1_inv = 1./(1e-6+diag1)
        diag2_inv = 1./(1e-6+diag2)
        diag1_inv = paddle.reshape(diag1_inv,[diag1_inv.shape[0],1])
        diag2_inv = paddle.reshape(diag2_inv,[diag2_inv.shape[0],1])
        pri_perp = I-ri*ri*paddle.matmul(ones,paddle.transpose(diag1_inv,[1,0]))
        prj_perp = I-rj*rj*paddle.matmul(ones,paddle.transpose(diag2_inv,[1,0]))
        val = paddle.sum(ri*prj_perp*ri, axis=0) + paddle.sum(rj*pri_perp*rj, axis=0)
        val = paddle.sqrt(val)
        return val

    def estimation_net_run(self, z):
        layer_en_1 = self.layer_en_1(z)
        layer_en_1 = self.activate(layer_en_1)
        layer_en_2 = self.layer_en_2(layer_en_1)
        layer_en_3 = F.softmax(layer_en_2)
        return layer_en_3

    def forward(self, data, fea):
        layer_1_ae_2, layer_1_ae_4 = self.compression_net_1_run(data)
        layer_2_ae_2, layer_2_ae_4 = self.compression_net_2_run(fea)
        loss_e_opd_1 = self.extract_feature(data,layer_1_ae_4)
        loss_e_opd_2 = self.extract_feature(fea, layer_2_ae_4)
        z_1 = paddle.concat([layer_1_ae_2,loss_e_opd_1*self.w_cat[0]], axis=1)
        z_2 = paddle.concat([layer_2_ae_2,loss_e_opd_2*self.w_cat[1]], axis=1)
        z = z_1 + z_2*self.w_add
        layer_en_3 = self.estimation_net_run(z)
        output = layer_en_3
        ae_layers = [layer_1_ae_4,layer_2_ae_4]
        return ae_layers,z,output

class myloss(paddle.nn.Layer):
    def __init__(self,w):
        super(myloss,self).__init__()
        self.w_loss = w
        self.min = paddle.to_tensor(1e-12).astype('float32')

    def gmm(self,z,gamma):
        #with paddle.no_grad():
        gamma_sum = paddle.sum(gamma, axis=0)
        phi = paddle.mean(gamma, axis=0)
        mu = paddle.matmul(paddle.transpose(gamma,perm=[1,0]),z) / (paddle.reshape(gamma_sum,[gamma_sum.shape[0],1]) + self.min)
        z_centered = paddle.sqrt((paddle.reshape(gamma,[gamma.shape[0],gamma.shape[1],1]) + self.min)) * (
                paddle.reshape(z,[z.shape[0],1,z.shape[1]]) - paddle.reshape(mu,[1,mu.shape[0],mu.shape[1]])
        )
        sigma = paddle.matmul(
            paddle.transpose(z_centered,perm=[1,2,0]),paddle.transpose(z_centered,perm=[1,0,2])
        ) / (paddle.reshape(gamma_sum,[gamma_sum.shape[0],1,1]) + self.min)
        self.sigma = sigma
        n_features = z.shape[1]
        min_vals = paddle.diag(paddle.ones([n_features])).astype('float32')*1e-6
        L = paddle.cholesky(sigma + paddle.reshape(min_vals,[1,min_vals.shape[0],min_vals.shape[1]]))

        z_centered_energy = paddle.reshape(z,[z.shape[0],1,z.shape[1]]) - paddle.reshape(mu,[1,mu.shape[0],mu.shape[1]])
        #print(L)
        L_inv = paddle.inverse(L)
        v = paddle.matmul(L_inv,paddle.transpose(z_centered_energy,[1,2,0]))
        # v = np.linalg.solve(L,paddle.transpose(z_centered_energy,[1,2,0]))
        L_diag = paddle.sum(paddle.triu(paddle.tril(L)),axis=1)
        log_det_sigma = 2.0*paddle.sum(paddle.log(L_diag),axis=1)
        d = z.shape[1]
        logits = paddle.log(paddle.reshape(phi,[phi.shape[0],1]))-0.5*(
                paddle.sum(paddle.square(v),axis=1) + d*paddle.log(paddle.to_tensor(2.0*np.pi))
                + paddle.reshape(log_det_sigma,[log_det_sigma.shape[0],1])
        )
        energy = -paddle.logsumexp(logits, axis=0)
        return energy

    def cov_diag(self):
        sigma_diag = paddle.sum(paddle.triu(paddle.tril(self.sigma)),axis=1)
        diag_loss = paddle.sum(1.0 / (sigma_diag + self.min))
        return diag_loss

    def forward(self, data, fea,ae_layers, z, soft_out):
        loss_mse_1 = F.mse_loss(data,ae_layers[0])
        loss_mse_2 = F.mse_loss(fea, ae_layers[1])
        energy = self.gmm(z,soft_out)
        loss_mse = self.w_loss[0][0] * loss_mse_1 + self.w_loss[0][1] * loss_mse_2
        loss_energy = self.w_loss[1] * paddle.mean(energy)
        cov_diag_loss = self.w_loss[2] * self.cov_diag()
        loss = loss_mse + loss_energy + cov_diag_loss
        return loss, paddle.reshape(energy,[energy.shape[0],1])


if __name__ == '__main__':
    config = {'sandiego_plane': [1e-1,1e-1],#[1e-1,1e0]
              'HYDICE_data': [1e-1,1e-1],#[1e-1,1e0]
              'abu-beach-3': [1e-1,1e-1],#[1e-1,1e0]
              'abu-urban-1': [1e-1,1e-1],#[1e-1,1e0]
              'abu-urban-2': [1e-1,1e-1],#[1e-1,1e-1]
              'abu-urban-4': [1e-1,1e-1]}#[1e0,1e-1]
              #'GrandIsle': [1e-1, 1e-1]}#[1e0, 1e0]
    #file_name = 'GrandIsle'
    #file_name = 'sandiego_plane'
    for file_name in config.keys():
        print('======>>>>>>dataset: {}'.format(file_name))
        #RES = open('res/parameter-sensitivity-analysis-a.txt','a')
        #RES2 = open('res/parameter-sensitivity-analysis-f.txt', 'a')
        #RES.write(file_name+'\n')
        #for lam2 in [1e-2,1e-1,1e0,1e1]:
        if True:
            auc_mean = []
            fpr_mean = []
            energies_mat = {}
            for i in range(10):
                path = 'datasets/'+file_name+'.mat'
                #path='conv2d-fea-test.mat'
                datasets = mydatasets(path)
                if datasets.num_sampales < 20000:
                    batch_size =int(datasets.num_sampales/1)
                    activate = 'tanh'
                else:
                    batch_size = 20000
                    activate = 'relu'
                epochs = 20
                num_clusters = 5
                hidden_node = 9
                hidden_nodes_1 = [200, hidden_node]
                hidden_nodes_2 = [30, hidden_node]
                est_hiddens = 60
                w_loss = [[1e0, config[file_name][0]], 1e-1, 1e-3]
                #w_loss = [[1e0, config[file_name][0]], lam2, 1e-3]
                w_cat = [1e0, 1e0]
                w_add = config[file_name][1]
                #w_add = a
                data_dim, fea_dim = datasets.get_dim()
                train_loader = paddle.io.DataLoader(datasets, batch_size=batch_size)
                liade2 = mymodel(num_clusters=num_clusters,
                                 hidden_nodes_1=hidden_nodes_1,
                                 hidden_nodes_2=hidden_nodes_2,
                                 est_hiddens=est_hiddens,
                                 data_dim=data_dim, fea_dim=fea_dim,
                                 w_cat=w_cat, w_add=w_add,activate=activate)
                liade2.train()
                optim = paddle.optimizer.Adam(learning_rate=1e-4, parameters=liade2.parameters())
                loss_fn = myloss(w=w_loss)

                loss_plot = []

                time_s = time.time()

                for epoch in range(epochs):
                    loss_epoch = 0
                    n = 0
                    energies = paddle.zeros([1, 1],dtype='float32')
                    for batch_id, loader in enumerate(train_loader()):
                        data = loader[0]
                        fea = loader[1]
                        ae_layers, z, output = liade2(data, fea)
                        loss, energy_back = loss_fn(data, fea, ae_layers, z, output)
                        '''
                        loss, energy_back = loss_fn(data.astype('float64'),fea.astype('float64'),
                                       [ae_layers[0].astype('float64'),ae_layers[1].astype('float64')],
                                       z.astype('float64'),output.astype('float64'))
                        '''
                        energies = paddle.concat([energies, energy_back], axis=0)
                        loss.backward()
                        loss_epoch += loss.numpy()
                        n += 1
                        optim.step()
                        optim.clear_grad()
                    #print('epoch'+str(epoch+1)+'/'+str(epochs)+':'+str(loss_epoch/n))
                    loss_plot.append(loss_epoch)

                time_e = time.time()
                print(time_e-time_s)

                energies = energies.numpy()[1:,:]
                # TODO np.clip  均值方差归一化
                # energies = (energies-energies.min()) / (energies.max()-energies.min())
                energies_ms = (energies - np.mean(energies)) / np.std(energies)
                gture, map_size = datasets.get_map()
                gture = np.reshape(gture, [map_size[0]*map_size[1], 1])
                auc = roc_auc_score(gture, energies)
                _, fpr = false_alarm_rate(gture, energies)
                print(str(i+1)+'  '+str(auc)+'  '+str(fpr))
                auc_mean.append(auc)
                fpr_mean.append(fpr)
                energies_mat.update({'res'+str(i): np.reshape(energies, map_size)})
            auc_mean = np.array(auc_mean)
            fpr_mean = np.array(fpr_mean)
            print(np.mean(auc_mean), np.mean(fpr_mean))
            #RES.write('num_clusters={}\n'.format(num_clusters))
            #RES.write('auc={:.6f}\n'.format(np.mean(auc_mean)))
            #RES.write('fpr={:.6f}\n'.format(np.mean(fpr_mean)))
            #RES.write('{:.6f} '.format(np.mean(auc_mean)))
            #RES2.write('{:.6f} '.format(np.mean(fpr_mean)))
        #RES.write('\n')
        #RES2.write('\n')
        #RES.close()
        #RES2.close()
