# Description of the PS (Planted Solution) problems:
# The s_i are Ising spins that can take on values \pm 1. The files themselves contain
# the J_ij and h_i parameter values where each line has the structure i j J_ij if the two indices are different
# or i i h_i if the first two entries are the same. The ground state to be reached is in the file name
# next to the character “e”. For example, in the file inst_p7_q11_e-70.txt the ground state energy is -70.

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import time
from multiprocessing import Process

RegCoeffArray = np.zeros((11, 4, 2))
RegCoeffArray[0, 0, :] = [2.0, 0.0]
RegCoeffArray[0, 1, :] = [0.2, 2.0]
RegCoeffArray[0, 2, :] = [0.5, 2.5]
RegCoeffArray[0, 3, :] = [1.0, 3.0]

RegCoeffArray[1, 0, :] = [2.0, 0.0]
RegCoeffArray[1, 1, :] = [0.2, 2.0]
RegCoeffArray[1, 2, :] = [1.0, 2.5]
RegCoeffArray[1, 3, :] = [1.5, 3.0]

RegCoeffArray[2, 0, :] = [2.0, 0.0]
# RegCoeffArray[2, 1, :] = [0.2, 1.5]
RegCoeffArray[2, 1, :] = [-0.5, -1.5]
RegCoeffArray[2, 2, :] = [0.6, 2.0]
RegCoeffArray[2, 3, :] = [1.5, 3.0]

RegCoeffArray[3, 0, :] = [2.0, 0.0]
RegCoeffArray[3, 1, :] = [0.2, 2.0]
RegCoeffArray[3, 2, :] = [1.0, 3.0]
RegCoeffArray[3, 3, :] = [1.5, 3.5]

RegCoeffArray[4, 0, :] = [1.4, 1.3]
RegCoeffArray[4, 1, :] = [0.2, 2.0]
RegCoeffArray[4, 2, :] = [1.0, 3.0]
RegCoeffArray[4, 3, :] = [1.5, 3.5]

RegCoeffArray[5, 0, :] = [1.7, 1.4]
RegCoeffArray[5, 1, :] = [0.1, 0.6]
RegCoeffArray[5, 2, :] = [0.5, 0.8]
RegCoeffArray[5, 3, :] = [0.4, 0.6]

RegCoeffArray[6, 0, :] = [2.1, 1.5]
RegCoeffArray[6, 1, :] = [0.2, 0.8]
RegCoeffArray[6, 2, :] = [0.2, 0.8]
RegCoeffArray[6, 3, :] = [0.3, 0.7]

RegCoeffArray[7, 0, :] = [1.6, 1.6]
RegCoeffArray[7, 1, :] = [0.0, 0.0]
RegCoeffArray[7, 2, :] = [0.0, 0.0]
RegCoeffArray[7, 3, :] = [0.0, 0.0]

RegCoeffArray[8, 0, :] = [1.7, 1.4]
RegCoeffArray[8, 1, :] = [0.0, 0.0]
RegCoeffArray[8, 2, :] = [0.0, 0.0]
RegCoeffArray[8, 3, :] = [0.0, 0.0]

RegCoeffArray[9, 0, :] = [1.7, 1.5]
RegCoeffArray[9, 1, :] = [0.0, 0.0]
RegCoeffArray[9, 2, :] = [0.0, 0.0]
RegCoeffArray[9, 3, :] = [0.0, 0.0]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class QUBOModel(torch.nn.Module):
    def __init__(self, Jh, model_type, filenum, tgt_engy, num_trials, RegCoeffArray4File):
        self.trial = 0
        self.tbwrite_en = 0
        self.reg_coeff = 0
        super(QUBOModel, self).__init__()
        self.num_trials = num_trials
        self.filenum = filenum
        self.tgt_engy = tgt_engy
        # self.tbwriter = SummaryWriter()
        self.noise_scale = 0.01
        self.perturb_scale = 0.01
        L = int(Jh.max())
        J = np.zeros((L, L), dtype=float)
        h = np.zeros((L, 1), dtype=float)
        for row_cnt in range(Jh.shape[0]):
            idx1 = Jh[row_cnt, 0] - 1
            idx2 = Jh[row_cnt, 1] - 1
            val = Jh[row_cnt, 2]
            if idx1 != idx2:
                J[idx1, idx2] = val
            else:
                h[idx1, 0] = val

        # print(J)
        # print(h)

        self.model_type = model_type
        if model_type in [0, 1]:    # Real and Complex
            self.psi_width = 1
        elif model_type == 2:                     # Sphere
            self.psi_width = 3
        elif model_type == 3:                     # Quaternion
            self.psi_width = 4

        self.L = L
        self.J = torch.tensor(J, dtype=torch.float32)
        self.h = torch.tensor(h, dtype=torch.float32)

        self.psi_angles = torch.nn.Parameter(2*torch.rand(L, self.psi_width)-1)

        if self.model_type == 1:  # Complex
            self.J = self.J.to(torch.complex64)
            self.h = self.h.to(torch.complex64)

        self.reg_coeff_array = np.zeros((4,2))
        for ii in range(4):
            self.reg_coeff_array[ii, :] = RegCoeffArray4File[ii,:]

    def get_loss(self, psi, apply_reg=1):
        loss = ((psi[:, 0:1].H @ self.J + self.h.T) @ psi[:, 0:1]).real
        for img_cnt in range(1, self.psi_width):
            loss += psi[:, img_cnt:img_cnt+1].H @ self.J @ psi[:, img_cnt:img_cnt+1]
        if apply_reg:
            if self.model_type == 0:
                loss_diag_reg = (psi[:, 0] ** 2).sum()
            elif self.model_type == 1:
                loss_diag_reg = (psi[:, 0].imag ** 2).sum()
            elif self.model_type in [2, 3]:
                loss_diag_reg = (psi[:, 1] ** 2).sum()
                for coordinate_cnt in range(1, self.psi_width):
                    loss_diag_reg += (psi[:, coordinate_cnt] ** 2).sum()
            else:
                print('wrong model type')
                exit()
            loss += self.reg_coeff * loss_diag_reg

        return loss

    def forward(self):
        if self.model_type == 0:   # Real
            return torch.cos(self.psi_angles * torch.pi)
        elif self.model_type == 1:  # Complex
            return torch.exp(1j * self.psi_angles * torch.pi)
        # elif self.model_type in [2,3]:  # Sphere or Quaternion
        #     return self.psi_angles
        elif self.model_type in [2]:  # Sphere
            tmp1 = torch.sin(self.psi_angles[:, 0:1] * torch.pi)
            X = tmp1 * torch.cos(self.psi_angles[:, 1:2] * torch.pi)
            Y = tmp1 * torch.sin(self.psi_angles[:, 1:2] * torch.pi)
            Z = torch.cos(self.psi_angles[:, 0:1] * torch.pi)
            return torch.cat([X, Y, Z], dim=1)
        elif self.model_type in [3]:  # Quaternion
            tmp1 = torch.cos(self.psi_angles[:, 0:1] * torch.pi)
            tmp2 = tmp1 * torch.cos(self.psi_angles[:, 1:2] * torch.pi)
            # X = torch.cos(self.psi_angles[:, 0:1] * torch.pi) * torch.cos(self.psi_angles[:, 1:2] * torch.pi) * torch.cos(self.psi_angles[:, 2:3] * torch.pi)
            # Y = torch.cos(self.psi_angles[:, 0:1] * torch.pi) * torch.cos(self.psi_angles[:, 1:2] * torch.pi) * torch.sin(self.psi_angles[:, 2:3] * torch.pi)
            # Z = torch.cos(self.psi_angles[:, 0:1] * torch.pi) * torch.sin(self.psi_angles[:, 1:2] * torch.pi)
            X = tmp2 * torch.cos(self.psi_angles[:, 2:3] * torch.pi)
            Y = tmp2 * torch.sin(self.psi_angles[:, 2:3] * torch.pi)
            Z = tmp1 * torch.sin(self.psi_angles[:, 1:2] * torch.pi)
            W = torch.sin(self.psi_angles[:, 0:1] * torch.pi)
            return torch.cat([X, Y, Z, W], dim=1)


    def reset_weights(self):
        self.psi_angles.data = 2 * torch.rand(self.L, self.psi_width) - 1

    def get_rounded_psi(self, psi):
        psi_ = psi.clone()
        psi_[psi_[:, 0].real >= 0, 0] = +1
        psi_[psi_[:, 0].real <= 0, 0] = -1
        psi_[:, 1:] = 0
        return psi_

    def train(self, epochs=10000, lr=0.01):
        if self.tbwrite_en == 1:
            tmp_tbwriter = SummaryWriter(f'runs/pq{self.filenum}/model{self.model_type}/try{self.trial}')

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # normalizer1 = WeightConstraint()
        with torch.no_grad():
            tmp = self.forward()
            Estart = self.get_loss(tmp)
        loss_vec = []
        loop = tqdm(range(epochs))
        for epoch in loop:
            if epoch == 0:
                self.reg_coeff = self.reg_coeff_array[self.model_type,0]
            elif epoch == epochs//2:
                self.reg_coeff = self.reg_coeff_array[self.model_type,1]
            optimizer.zero_grad()
            psi = self.forward()
            loss = self.get_loss(psi)


            # self.tbwriter.add_scalar("Loss/train", loss, epoch)
            # run from command line:     tensorboard --logdir  runs
            loss.backward()
            # Add Noise
            for param in self.parameters():
                if param.grad is not None:
                    param.grad += torch.randn_like(param.grad) * self.noise_scale
            optimizer.step()
            if epoch % 100 == 0:
                for param in self.parameters():
                    param.data += torch.randn_like(param.data) * self.perturb_scale

            # if self.model_type in [2, 3]:  # Constrain
            #     self.apply(normalizer1)

            loss_continuous_final = loss.item()
            loss_vec.append(loss_continuous_final)
            with torch.no_grad():
                psi = self.forward().detach().clone()
                rounded_psi = self.get_rounded_psi(psi)
                Eround = self.get_loss(rounded_psi, apply_reg=0)
            loop.set_description('Estart:{:.4f}, Eend:{:.4f}, Eround:{:.4f}'.format(Estart.item(), loss.item(), Eround.item()))
            if Eround.item() == self.tgt_engy:
                np.savetxt('ps_res/model'+str(self.model_type)+'_ans_pq'+str(self.filenum)+'_'+str(self.num_trials)+'trials.txt', rounded_psi.real, fmt='%d')

            if epoch == epochs//2-1:
                psi_avg_engy_0 = (psi.numpy()[:, 0].real ** 2).mean()
                E0 = Eround.item()
                psi_0 = psi.detach().clone().numpy()
            elif epoch == epochs-1:
                psi_avg_engy_1 = (psi.numpy()[:, 0].real ** 2).mean()
                E1 = Eround.item()
                psi_1 = psi.detach().clone().numpy()
            if self.tbwrite_en == 1:
                tmp_tbwriter.add_scalar('Loss', loss_continuous_final, global_step=epoch)
                tmp_tbwriter.add_scalar('EPsi', psi_engy_0, global_step=epoch)
                tmp_tbwriter.flush()
            psi_fin = psi.detach().clone().numpy()
        return E0, E1, psi_0, psi_1, psi_avg_engy_0, psi_avg_engy_1


def run_search(proc_num, NModelTypes, model_type_enable, ising_arr, file_num, min_energy, trials_num, nepocs, lr, trg_engy):
    engy_fname = 'ps_res/proc{:d}_engy_pq{:d}_{:d}trials_{:d}ep.npy'.format(proc_num, file_num, trials_num, nepocs)
    if os.path.exists(engy_fname):
        engy_array = np.load(engy_fname)
    else:
        engy_array = np.zeros((NModelTypes, trials_num,5))
    for mtype_cnt in range(NModelTypes):
        if model_type_enable[mtype_cnt] == 1:
            mymodel = QUBOModel(ising_arr, mtype_cnt, file_num, min_energy, trials_num, RegCoeffArray[file_num,:,:])
            success_num = 0
            for cnt in range(trials_num):
                torch.manual_seed(cnt*6+proc_num)
                mymodel.reset_weights()
                if 0:  #cnt < 20:
                    mymodel.tbwrite_en = 1
                else:
                    mymodel.tbwrite_en = 0

                mymodel.trial = cnt
                engy0, engy1, psi_final_0, psi_final_1, EPsi0, EPsi1 = mymodel.train(nepocs, lr)
                if engy1 == trg_engy:
                    success_num += 1

                engy_array[mtype_cnt, cnt, 0] = int(engy0)
                engy_array[mtype_cnt, cnt, 1] = int(engy1)
                engy_array[mtype_cnt, cnt, 2] = float(EPsi0)
                engy_array[mtype_cnt, cnt, 3] = float(EPsi1)
                print('\n Proc {:d}: pq{:d}, model_type={:d}, cnt={:d}/{:d}: energy = {:.1f},{:.1f}'.format( proc_num, file_num, mtype_cnt, cnt + 1,
                                                                                          trials_num, engy0, engy1), np.min(engy_array[:,:,0], axis=1), np.min(engy_array[:,:,1], axis=1),
                      np.average(engy_array[:,:cnt+1,2], axis=1).round(2),np.average(engy_array[:,:cnt+1,3], axis=1).round(2))
            np.save(engy_fname, engy_array)

if __name__ == '__main__':
    ############# CONTROL Block: Set The Parameters Here
    multi_proc_num = 1  # to work with multiprocessing increase this number (e.g. 6)
    lr = 0.003
    # ntrials = 1000
    ntrials = 10
    nep = 2000
    run_enable = 1
    show_results = 1
    show_avg_spin_pwr = 0

    # files_enable = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    files_enable = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # model_type_enable = [1, 1, 1, 1]   # 0-Real, 1-Complex, 2-Sphere, 3-Quaternions
    model_type_enable = [1, 0, 0, 0]   # 0-Real, 1-Complex, 2-Sphere, 3-Quaternions
    method_names = ['REAL', 'COMPLEX', 'SPHERE', 'QUAT']
    ####################################################
    print_hi('PS')
    if not os.path.isdir('ps_res'):
        os.mkdir('ps_res')
    if not os.path.isdir('ps_params'):
        os.mkdir('ps_params')
    # if not os.path.isdir('plots'):
    #     os.mkdir('plots')
    inputfiles = os.listdir('./ps_data')
    nfiles = len(inputfiles)
    print('ntrials={:d} nep={:d} lr={:f}'.format(ntrials, nep, lr))
    NModelTypes = len(model_type_enable)
    if 1:  # Init.
        problem_sizes = np.zeros(nfiles, dtype=int)
        file_cnt = -1
        for data_fname in inputfiles:
            file_cnt += 1
            data_array = np.loadtxt('./ps_data/' + data_fname, dtype=int)
            problem_sizes[file_cnt] = int(data_array[:, 0].max())
        sort_idx = np.argsort(problem_sizes).T
        inputfiles_sorted = [inputfiles[file_idx] for file_idx in sort_idx]
        print(inputfiles_sorted)
        np.savetxt('ps_params/inputfiles_sorted.txt', inputfiles_sorted, fmt='%s')
        file_cnt = -1
        for data_fname in inputfiles_sorted:   # Check sorting
            file_cnt += 1
            data_array = np.loadtxt('./ps_data/' + data_fname, dtype=int)
            problem_sizes[file_cnt] = int(data_array[:, 0].max())
        min_energies = [int(data_fname.split('_e')[-1].split('.')[0]) for data_fname in inputfiles_sorted]
        print(np.vstack([problem_sizes, min_energies]))
        np.savetxt('ps_params/problem_sizes.txt', problem_sizes, fmt='%d')
        np.savetxt('ps_params/min_energies.txt', min_energies, fmt='%d')

    if run_enable:
        inputfiles_sorted = np.loadtxt('ps_params/inputfiles_sorted.txt', dtype=str)
        problem_sizes = np.loadtxt('ps_params/problem_sizes.txt')
        min_energies = np.loadtxt('ps_params/min_energies.txt')
        t0 = time.time()
        for nfile in range(nfiles):
            trg_engy =  min_energies[nfile]
            if files_enable[nfile] == 1:
                t1 = time.time()
                fname = inputfiles_sorted[nfile]
                ising_array = np.loadtxt('./ps_data/' + fname, dtype=int)

                tasks = []
                for proc_num in range(1,multi_proc_num):
                    tasks.append(Process(target=run_search,
                                         args=(proc_num, NModelTypes, model_type_enable, ising_array, nfile, min_energies[nfile], ntrials, nep, lr, trg_engy)))
                for thread in tasks:
                    thread.start()

                run_search(0, NModelTypes, model_type_enable, ising_array, nfile, min_energies[nfile], ntrials, nep, lr, trg_engy)

                for thread in tasks:
                    thread.join()

                print('time=', time.time() - t1)
        print('time total=', time.time() - t0)

    if show_results:
        inputfiles_sorted = np.loadtxt('ps_params/inputfiles_sorted.txt', dtype=str)
        problem_sizes = np.loadtxt('ps_params/problem_sizes.txt')
        for nfile in range(nfiles):
            proc_num = 0
            if files_enable[nfile]:
                for proc_num in range(multi_proc_num):
                    engy_fname = 'ps_res/proc{:d}_engy_pq{:d}_{:d}trials_{:d}ep.npy'.format(proc_num, nfile, ntrials, nep)
                    engy_array_proc = np.load(engy_fname)
                    if proc_num == 0:
                        engy_array_0 = engy_array_proc[:, :, 0]
                        engy_array_1 = engy_array_proc[:, :, 1]
                        epsi_array_0 = engy_array_proc[:, :, 2]
                        epsi_array_1 = engy_array_proc[:, :, 3]
                    else:
                        engy_array_0 = np.hstack([engy_array_0, engy_array_proc[:, :, 0]])
                        engy_array_1 = np.hstack([engy_array_1, engy_array_proc[:, :, 1]])
                        epsi_array_0 = np.hstack([epsi_array_0, engy_array_proc[:, :, 2]])
                        epsi_array_1 = np.hstack([epsi_array_1, engy_array_proc[:, :, 3]])

                res_min_engy_0 = [int(engy_array_0[n].min()) for n in range(NModelTypes)]
                res_min_engy_1 = [int(engy_array_1[n].min()) for n in range(NModelTypes)]
                print(res_min_engy_0, res_min_engy_1)
                nbins = 100

                if sum(model_type_enable) > 1:

                    fig1, (ax12, ax34) = plt.subplots(2, 2, sharex=False, sharey=False)
                    ax12[0].hist(engy_array_1[0, :], nbins, label='REAL\nMin={:d}({:d})'.format(res_min_engy_1[0],sum(engy_array_1[0, :]==res_min_engy_1[0])))
                    ax12[1].hist(engy_array_1[1, :], nbins, label='COMPLEX\nMin={:d}({:d})'.format(res_min_engy_1[1],sum(engy_array_1[1, :]==res_min_engy_1[1])))
                    ax34[0].hist(engy_array_1[2, :], nbins, label='SPHERE\nMin={:d}({:d})'.format(res_min_engy_1[2],sum(engy_array_1[2, :]==res_min_engy_1[2])))
                    ax34[1].hist(engy_array_1[3, :], nbins, label='QUAT\nMin={:d}({:d})'.format(res_min_engy_1[3],sum(engy_array_1[3, :]==res_min_engy_1[3])))
                    ax12[0].legend(loc='upper right')
                    ax12[1].legend(loc='upper right')
                    ax34[0].legend(loc='upper right')
                    ax34[1].legend(loc='upper right')
                    xlim_0 = min(ax12[0].get_xlim(), ax12[1].get_xlim(), ax34[0].get_xlim(), ax34[1].get_xlim())[0]
                    xlim_1 = max(ax12[0].get_xlim(), ax12[1].get_xlim(), ax34[0].get_xlim(), ax34[1].get_xlim())[1]
                    ax12[0].set_xlim([xlim_0, xlim_1])
                    ax12[1].set_xlim([xlim_0, xlim_1])
                    ax34[0].set_xlim([xlim_0, xlim_1])
                    ax34[1].set_xlim([xlim_0, xlim_1])
                    title_str = '(#{:d}) {:s} \n[Sz={:d}, NTrials={:d}, Nep={:d}, lr={:.3f}]'.format(nfile+1,
                                inputfiles_sorted[nfile], int(problem_sizes[nfile]), ntrials * multi_proc_num, nep, lr)
                    fig1.suptitle(title_str)
                    fig1.show()

                    if show_avg_spin_pwr:
                        fig2, (ax12, ax34) = plt.subplots(2, 2, sharex=False, sharey=False)
                        ax12[0].hist(epsi_array_0[0, :], nbins, label='REAL 0')
                        ax12[1].hist(epsi_array_0[1, :], nbins, label='COMPLEX 0')
                        ax34[0].hist(epsi_array_0[2, :], nbins, label='SPHERE 0')
                        ax34[1].hist(epsi_array_0[3, :], nbins, label='QUAT 0')
                        ax12[0].hist(epsi_array_1[0, :], nbins, label='REAL 1')
                        ax12[1].hist(epsi_array_1[1, :], nbins, label='COMPLEX 1')
                        ax34[0].hist(epsi_array_1[2, :], nbins, label='SPHERE 1')
                        ax34[1].hist(epsi_array_1[3, :], nbins, label='QUAT 1')
                        ax12[0].legend(loc='upper right')
                        ax12[1].legend(loc='upper right')
                        ax34[0].legend(loc='upper right')
                        ax34[1].legend(loc='upper right')
                        ax12[0].set_xlim([0, 1])
                        ax12[1].set_xlim([0, 1])
                        ax34[0].set_xlim([0, 1])
                        ax34[1].set_xlim([0, 1])
                        title_str = '(#{:d}) {:s} \n[Sz={:d}, NTrials={:d}, Nep={:d}, lr={:.3f}]'.format(nfile+1,
                                    inputfiles_sorted[nfile], int(problem_sizes[nfile]), ntrials * multi_proc_num, nep, lr)
                        fig2.suptitle(title_str)
                        fig2.show()

                else:
                    for model_type_cnt in range(NModelTypes):
                        if model_type_enable[model_type_cnt] == 1:
                            fig = plt.figure()
                            res_min_engy_cnt_0 = np.sum((engy_array_0[model_type_cnt, :] == res_min_engy_0[model_type_cnt]))
                            res_min_engy_cnt_1 = np.sum((engy_array_1[model_type_cnt, :] == res_min_engy_1[model_type_cnt]))
                            leg_label0 = 'Stage0: Min=' + str(res_min_engy_0[model_type_cnt]) + ' (' + str(res_min_engy_cnt_0) + ')'
                            plt.hist(engy_array_0[model_type_cnt, :], 100, label=leg_label0)
                            leg_label1 = 'Stage1: Min=' + str(res_min_engy_1[model_type_cnt]) + ' (' + str(res_min_engy_cnt_1) + ')'
                            plt.hist(engy_array_1[model_type_cnt, :], 100, label=leg_label1)
                            plt.legend(loc='upper left')
                            plt.ylabel(method_names[model_type_cnt])
                            # plt.ylim([0, 50])
                            title_str = '(#{:d}) {:s}, KReg=[{:.1f},{:.1f}] \n[Sz={:d}, NTrials={:d}, Nep={:d}, lr={:.3f}]'.format(nfile+1,
                                inputfiles_sorted[nfile], RegCoeffArray[nfile,model_type_cnt,0], RegCoeffArray[nfile,model_type_cnt,1], int(problem_sizes[nfile]), ntrials * multi_proc_num, nep, lr)
                            fig.suptitle(title_str)
                            plt.xlabel('Minimal Energy Found')
                            plt.ylabel('# Trials')
                            fig.show()

                    if show_avg_spin_pwr:
                        for model_type_cnt in range(NModelTypes):
                            if model_type_enable[model_type_cnt] == 1:
                                fig2 = plt.figure()
                                plt.hist(epsi_array_0[model_type_cnt, :], nbins, label=method_names[model_type_cnt]+' 0')
                                plt.hist(epsi_array_1[model_type_cnt, :], nbins, label=method_names[model_type_cnt]+' 1')
                                plt.legend(loc='upper right')
                                plt.xlim([0, 1])
                                title_str = ('(#{:d}) {:s}, KReg=[{:.1f},{:.1f}] \n[Sz={:d}, NTrials={:d}, Nep={:d}, lr={:.3f}]'
                                     .format(nfile+1, inputfiles_sorted[nfile], RegCoeffArray[nfile,model_type_cnt,0], RegCoeffArray[nfile,model_type_cnt,1], int(problem_sizes[nfile]), ntrials * multi_proc_num, nep, lr))
                                fig2.suptitle(title_str)
                                plt.xlabel(r'$E|spin|^2$')
                                plt.ylabel('# Trials')
                                fig2.show()
