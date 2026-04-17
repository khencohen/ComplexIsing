import os
import torch
from tqdm import tqdm
import numpy as np

def my_det(A):
    if A.shape[0] < A.shape[1]:
        det_out = np.linalg.det(np.matmul(A, A.T))
    else:
        det_out = np.linalg.det(np.matmul(A.T, A))
    return det_out

def normalize_matrix_det(A):
    M, N = A.shape[0], A.shape[1]
    detAA = my_det(A)
    assert detAA != 0
    while np.isinf(detAA):
        A = A/1.0001
        detAA = my_det(A)
    B = A / detAA ** (0.5 / min(M, N))
    assert abs(my_det(B) - 1) < 1E-2
    return B

class Axb:
    def __init__(self, N, M, Cardi=0.5, nbits=1, noise_std=0, complex_mode=0):
        '''
        Initialize the Axb problem, create messages m, matrix A, and vector b.
        :param N: number of matrix columns
        :param M: number of rows
        :param Cardi: Cardinality of the message vector
        '''

        self.N = N
        self.M = M
        self.Cardi = Cardi
        self.A = self.get_A()

        self.A = normalize_matrix_det(self.A)
        det = np.sqrt(np.linalg.det(np.matmul(self.A, self.A.T)))
        if np.abs(det - 1) > 1e-3:
            Warning('Matrix A is not normalized: det(A) = {}'.format(det))

        self.m = self.get_m(nbits)
        self.e = self.get_e(noise_std)
        self.b = self.get_b(self.m, self.e)

        return

    def get_A(self):
        return np.random.randn(self.M, self.N)

    def get_m(self, nbits):
        if nbits == 1:
            m = np.zeros((self.N, 1), dtype=np.int8)
            m[np.random.permutation(self.N)[:self.Cardi]] = 1
        elif nbits == 2:
            m = np.zeros((self.N, 1), dtype=np.int8)
            m[np.random.permutation(self.N)[:self.Cardi]] = np.random.randint(1, 4, size=(self.Cardi, 1))
        return m

    def get_e(self, err_sig=0.0):
        err = err_sig * np.random.randn(self.M, 1)
        return err

    def get_b(self, m, err=None, err_sig=0.0):
        if err is None:
            err = self.get_e(err_sig=err_sig)
        return np.matmul(self.A, m) + err

class ComplexNet(torch.nn.Module):
    '''
    This class is used to evaluate the value of n complex numbers
    '''

    def __init__(self, A, m, e, b, nbits=1, model_type=0, regtype=0, nep_switch_to_PB=0, init_weights=[0], tb_enable=0, avg_cnt=-1, noise_std=-1):
        super(ComplexNet, self).__init__()
        self.eliptic_ratio = 1
        self.regtype = regtype
        self.nbits = nbits
        # Define A
        self.N = A.shape[1]
        self.M = A.shape[0]
        self.A = torch.tensor(A, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.e = torch.tensor(e, dtype=torch.float32)
        self.m = torch.tensor(m*2-1, dtype=torch.float32)
        # M1, M2 = get_hami_2(A.numpy(), b.numpy(), 1)
        # W1 = np.matmul(A.numpy().T, A.numpy())
        # W2 = -2 * np.matmul(A.numpy().T, b.numpy())

        # self.M1 = torch.tensor(M1, dtype=torch.complex64)
        # self.M2 = torch.tensor(M2, dtype=torch.complex64)
        W1 = np.matmul(A.T, A)
        self.W1 = torch.tensor(W1, dtype=torch.float32)
        self.W2 = torch.tensor(-2 * np.matmul(A.T, b), dtype=torch.float32)
        # 2 (A(x+1)/2-b)  = Ax - (2b-sum(A,axis=1))
        self.W2_shifted = torch.tensor(-2 * np.matmul(A.T, 2*b-np.sum(A,axis=1).reshape((-1,1))), dtype=torch.float32)
        self.W1diag = torch.tensor(np.atleast_2d(np.diag(W1)).T, dtype=torch.float32)
        self.W1_rmdiag = torch.tensor(W1 - np.diag(np.diag(W1)), dtype=torch.float32)
        # self.W1_rmdiag = torch.tensor(W1-np.diag(np.min(np.diag(W1))*np.ones((1,self.N))), dtype=torch.float32)
        # self.W1_rmdiag = torch.tensor(W1-np.diag(np.min(np.diag(W1)) * np.ones((self.N))), dtype=torch.float32)
        self.W1_adddiag = torch.tensor(W1 - 1.0*np.eye(self.N), dtype=torch.float32)
        self.W1_invdiag = torch.tensor(W1 - np.diag(2*(np.mean(np.diag(W1))-np.diag(W1))), dtype=torch.float32)
        self.model_type = model_type
        self.diag_reg_factor = 1.0
        # np.savetxt('diag_reg_factor.txt', np.atleast_1d(self.diag_reg_factor))

        self.diag_correction_en = 0
        self.diag_reg_offset = 5000

        if self.model_type in [0]:  # Real
            self.freedom_degree = 1
            self.psi_angles = torch.nn.Parameter(2*torch.rand(self.N, 1)-1)
        elif self.model_type in [1]:  # Complex
            self.A = self.A.to(torch.complex64)
            self.b = self.b.to(torch.complex64)
            self.e = self.e.to(torch.complex64)
            self.m = self.m.to(torch.complex64)
            self.W1 = self.W1.to(torch.complex64)
            self.W2 = self.W2.to(torch.complex64)
            self.W2_shifted = self.W2_shifted.to(torch.complex64)
            self.W1diag = self.W1diag.to(torch.complex64)
            self.W1_rmdiag = self.W1_rmdiag.to(torch.complex64)
            self.W1_adddiag = self.W1_adddiag.to(torch.complex64)
            self.freedom_degree = self.nbits

            if len(init_weights) == 1:
                self.psi_angles = torch.nn.Parameter(2*torch.rand(self.N, self.nbits)-1)
            else:
                self.psi_angles = torch.nn.Parameter(init_weights)
        elif self.model_type == 2:  # Sphere
            # Extend to 3 size vectors
            self.b = torch.cat([self.b, torch.zeros(self.M, 2)], dim=1)
            self.e = torch.cat([self.e, torch.zeros(self.M, 2)], dim=1)
            self.m = torch.cat([self.m, torch.zeros(self.N, 2)], dim=1)

            # self.psi_vec = torch.nn.Parameter(torch.randn(self.N, 4))
            self.psi_angles = torch.nn.Parameter(2*torch.rand(self.N, 2)-1)
            self.freedom_degree = 2
        elif self.model_type == 3:  # Quat
            # Extend to 4 size vectors
            self.b = torch.cat([self.b, torch.zeros(self.M, 3)], dim=1)
            self.e = torch.cat([self.e, torch.zeros(self.M, 3)], dim=1)
            self.m = torch.cat([self.m, torch.zeros(self.N, 3)], dim=1)

            # self.psi_vec = torch.nn.Parameter(torch.randn(self.N, 4))
            self.psi_angles = torch.nn.Parameter(2*torch.rand(self.N, 3)-1)
            self.freedom_degree = 3
        else:
            assert 0, 'Wrong model_type='+str(self.model_type)

    def reset_weights(self):
        self.psi_angles.data = 2 * torch.rand(self.N, self.freedom_degree) - 1
        pass

    def set_weights(self, ww):
        self.psi_angles.data = torch.tensor(ww[0:self.N, 0:self.freedom_degree],dtype=torch.float32)
        pass

    # x = (s + 1)/2  =>  s=-1 <=> x=0;  s=+1 <=> x=1;
    def get_loss(self, psi, epoch_num=0):
        if os.path.exists('diag_reg_factor.txt'):
            diag_reg_factor = np.loadtxt('diag_reg_factor.txt')
            if len(np.atleast_1d(diag_reg_factor)) == 1:
                self.diag_reg_factor = diag_reg_factor
        if epoch_num>self.diag_reg_offset:
            diag_reg_factor_current = self.diag_reg_factor
        else:
            diag_reg_factor_current = 0
        Kreg = torch.tensor(diag_reg_factor_current)
        RegMat = torch.tensor(diag_reg_factor_current * np.eye(self.N), dtype=torch.float32)
        if self.model_type in [0]:  # Real
            loss_diag_reg = - (psi[:, :1].real.H @ RegMat @ psi[:, :1].real)[0][0]
        elif self.model_type in [1]:  # Complex
            loss_diag_reg = + (psi[:, :1].imag.H @ RegMat @ psi[:, :1].imag)[0][0]
        elif self.model_type in [2, 3]:  # Sphere, Quaternion
            loss_diag_reg = + (psi[:, 1:2].H @ psi[:, 1:2])
            for ii in range(self.freedom_degree-1):
                loss_diag_reg = loss_diag_reg + (psi[:, 2+ii:3+ii]**2).sum()
            loss_diag_reg = Kreg * loss_diag_reg[0][0]
        else:
            print('wrong model_type')
            exit(1)
        if psi.shape[1]==1:
            loss = torch.real(psi[:, :1].H @ (self.W1 @ psi[:, :1] + self.W2_shifted))[0][0] + loss_diag_reg
        else:
            psi_shifted = (psi + torch.cat([torch.ones((self.N, 1)), torch.zeros((self.N, psi.shape[1] - 1))], dim=1)) / 2
            loss = 4 * torch.square(torch.abs(self.A @ psi_shifted - self.b)).sum() + loss_diag_reg  # mult by 4 to compensate factor 1/2 of psi
        return loss


# In the following derivation constants are dropped as soon as they appear.
# Ep{xT W1 x + xT W2} =
# SUM_ij W1ij[pi*pj + (1-pi)(1-pj) - pi(1-pj) - (1-pi)pj] + SUM_i W2i[pi - (1-pi)] + correction =
# = SUM_ij W1ij[2(2pi*pj-pi-pj)] + SUM_i W2i[2pi] + correction =
#    | correction = SUM_i W1ii[1-4(pi^2-pi)] = SUM_i W1ii[4(pi-pi^2)]
# = 4pT*W1*p + 2pT[-(W1+W1T)(1) + W2 + 2diag(W1)] - 4*(p^2T)diag(W1)
#    | q=2p-1 => p=(q+1)/2
# = (qT+1)W1(q+1) + (qT+1)[-(W1+W1T)(1) + W2 + 2diag(W1)] - (q+1)^2T*diag(W1) =
# = (qT)W1(q) + (qT)(W1+W1T)(1) + qT[-(W1+W1T)(1) + W2 + 2diag(W1)] - (q^2T)diag(W1) - (2qT)diag(W1) =
# = (qT)W1(q) + (qT)W2 - (q^2T)diag(W1)

    def get_rounded_psi(self, psi):
        # psi_rnd = psi
        psi_rnd = psi.clone()
        if self.model_type in [0, 1]: # Real, Complex
            psi_rnd[abs(psi_rnd.real-1) <= 1] = +1
            psi_rnd[abs(psi_rnd.real+1) <= 1] = -1
            if self.nbits == 2:
                psi_rnd[psi_rnd.real >= 2] = +3
                psi_rnd[psi_rnd.real < -2] = -3
            psi_rnd = psi_rnd.real
        else:  # Sphere, Quat
            psi_rnd[abs(psi_rnd.real-1) <= 1] = +1
            psi_rnd[abs(psi_rnd.real+1) <= 1] = -1
            psi_rnd = torch.cat([psi_rnd[:, 0:1], torch.zeros(self.N, psi_rnd.shape[1]-1)], dim=1)
        return psi_rnd

    def forward(self):
        if self.model_type == 0:   # Real
            return (self.nbits*2-1)*torch.cos(self.psi_angles * torch.pi)
        elif self.model_type == 1:  # Complex
            if self.nbits == 1:
                return torch.exp(1j * self.psi_angles[:, 0:1] * torch.pi)
            elif self.nbits == 2:
                return (2+torch.cos(self.psi_angles[:, 1:2] * torch.pi))*torch.exp(1j * self.psi_angles[:, 0:1] * 1 * torch.pi)
        elif self.model_type == 2:  # Sphere
            X = torch.sin(self.psi_angles[:, 0:1] * torch.pi) * torch.cos(self.psi_angles[:, 1:2] * torch.pi)
            Y = torch.sin(self.psi_angles[:, 0:1] * torch.pi) * torch.sin(self.psi_angles[:, 1:2] * torch.pi)
            Z = torch.cos(self.psi_angles[:, 0:1] * torch.pi)
            return torch.cat([X, Y, Z], dim=1)
        elif self.model_type == 3:  # Quat
            X = torch.cos(self.psi_angles[:, 0:1] * torch.pi) * torch.cos(self.psi_angles[:, 1:2] * torch.pi) * torch.cos(self.psi_angles[:, 2:3] * torch.pi)
            Y = torch.cos(self.psi_angles[:, 0:1] * torch.pi) * torch.cos(self.psi_angles[:, 1:2] * torch.pi) * torch.sin(self.psi_angles[:, 2:3] * torch.pi)
            Z = torch.cos(self.psi_angles[:, 0:1] * torch.pi) * torch.sin(self.psi_angles[:, 1:2] * torch.pi)
            W = torch.sin(self.psi_angles[:, 0:1] * torch.pi)
            return torch.cat([X, Y, Z, W], dim=1)
        else:
            assert 0, 'Wrong model type'


    def train(self, epochs=1000, lr=0.01, lasso_reg_factor=0.0, show=False, descript_enable=True, noise_std=0, avg=-1, ic=-1):
        zero_point = 1 - 2*self.nbits
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        with torch.no_grad():
            Emin = self.get_loss(self.m)

        loss_vec = []
        loop = tqdm(range(epochs))
        ep_cnt = 0
        for epoch in loop:
            ep_cnt += 1
            # Calculate the current loss
            optimizer.zero_grad()

            psi = self.forward()
            loss = self.get_loss(psi, epoch)

            if lasso_reg_factor > 0:
                if self.regtype == 0:
                    if psi.shape[1] == 1:
                        loss += lasso_reg_factor * torch.abs(psi - zero_point).sum()
                    else:
                        loss += lasso_reg_factor * torch.sqrt(torch.square(torch.abs(torch.cat(tensors = [psi[:, 0:1] - zero_point, psi[:, 1:]], axis = 1))).sum(axis=1)).sum()
                else:  # unused
                    loss += lasso_reg_factor * torch.abs((psi - zero_point).sum())
            # Zero the gradients
            loss.backward()
            # Update the weights
            optimizer.step()

            loss_vec.append(loss.item())

            with torch.no_grad():
                psi = self.forward().detach().clone()
                rounded_psi = self.get_rounded_psi(psi)
                if self.model_type in [1]:  # Complex
                    rounded_psi = rounded_psi + 0j
                E0 = self.get_loss(rounded_psi, epoch)
            psi_engy = (psi.numpy()[:, 0].real ** 2).mean()
            num_errors = sum(rounded_psi.numpy() != self.m.numpy())[0]

            # if ep_cnt % 100 == 0:
            #     data = np.concat([np.atleast_2d([self.model_type, float(self.diag_reg_factor), ep_cnt, int(num_errors)]).T, np.atleast_2d(psi[:,:1].real)])
            #     np.savetxt('./training_data.txt', data, delimiter=',')
            if descript_enable:
                loop.set_description('E:{:.4f}, Emin:{:.4f}, Loss:{:.4f}'.format(E0, Emin.item(), loss.item()))

        with torch.no_grad():
            psi = self.forward()

            psi = self.get_rounded_psi(psi)

            acc = torch.mean((torch.abs(psi - self.m) == 0).float())
            num_nonzeros = sum(psi.real.numpy() != zero_point)[0]
            num_errors = sum(psi.numpy() != self.m.numpy())[0]

            return acc.numpy(), num_nonzeros, num_errors, loss.item(), E0

def compute_loss(net, inputs, targets):
    outputs = net(inputs)
    return net.loss_fn(outputs, targets)


