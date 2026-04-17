from qubo_run_func import run_experiment
import os

import numpy as np
import time
from multiprocessing import set_start_method
from matplotlib import pyplot as plt

method_names = ['Real', 'Complex', 'Sphere', 'Quaternion']
lr = 0.001
model_types_enable = [1, 1, 1, 1]
use_multi_threading = 1
stop_if_0_errors = 0
run_mode = 2   # 0-run, 1-plot, 2-both
noise_std_list = np.array([0.20, 0.25, 0.30, 0.35])
opt_sol_en = 0
plot_mode = 0

testparams = np.zeros((1000, 100))

NParams = 13   # QUBO 160x160
# CARDINALITY              avg  nic  epochs rt  nbits  reg    N      M     tbd  Cardi rerun regen  rmdiag
testparams[1, :NParams] = [4,   5,   10000, 0,  1,     0,     160,   160,  -1,  -1,   1,    0,     0]
testparams[2, :NParams] = [50,  20,  10000, 0,  1,     0,     160,   160,  -1,  -1,   1,    0,     0]

NParams = 10   # Sparse Coding
# CARDINALITY              avg  nic  epochs rt  nbits  reg    N      M     tbd  Cardi rerun regen  rmdiag
testparams[3, :NParams] = [6,   10,  10000, 0,  1,     0.035, 16,    8,    -1,   3]
testparams[4, :NParams] = [100, 100, 10000, 0,  1,     0.035, 16,    8,    -1,   3]
testparams[5, :NParams] = [100, 100, 10000, 0,  1,     0.035, 32,    16,   -1,   6]
testparams[6, :NParams] = [20,  20,  10000, 0,  1,     0.035, 160,   80,   -1,   30]

tests_run_list = [2]

for tid in tests_run_list:
    if testparams[tid, 6] > 32:
        assert opt_sol_en == 0

def create_folders(folder_names):
    for fldname in folder_names:
        if not os.path.isdir(fldname):
            os.mkdir(fldname)


def plot_results(tid, N, M, name, noise_std_list, avg, nic, nep, cardi, model_types_enable, method_names , show=True, plot_ls=False, plot_mode=0):
    plot_avg, plot_nic = avg, nic
    nmethods = len(model_types_enable)
    y_max = 5
    plot_strings = ['pr-', 'vg-', '>b-', 'om-', '*y-','sk-', 'og--', 'sb--','<m--','*k--','ok-','sr-','pk-','.r-']
    avg_opt_nerr = np.zeros((len(noise_std_list)))+np.nan
    avg_nerr_list_min_nerr = np.zeros((len(noise_std_list), nmethods))+np.nan
    avg_nerr_list_min_engy = np.zeros((len(noise_std_list), nmethods))+np.nan
    avg_freg_list_avg = np.zeros((len(noise_std_list), nmethods, nic))+np.nan
    avg_freg_list = np.zeros((len(noise_std_list), nmethods, avg, nic))+np.nan
    noise_cnt = -1
    for noise_std in noise_std_list:
        noise_cnt += 1
        res_current_noise_point_fname = 'data/test{:d}/res_{:s}_noise{:.3f}.npy'.format(tid,name,noise_std)
        opt_current_noise_point_fname = 'data/test{:d}/opt_noise{:.3f}.npy'.format(tid,noise_std)
        if os.path.exists(res_current_noise_point_fname):
            tmp_res = np.load(res_current_noise_point_fname)[:,:plot_avg, :plot_nic, :]
            opt_res = np.load(opt_current_noise_point_fname)
            avg_opt_nerr[noise_cnt] = opt_res[:plot_avg, 0].mean()
            nerr, nones, freg, L1, L0 = tmp_res[:, :, :, 0], tmp_res[:, :, :, 1], tmp_res[:, :, :, 2], tmp_res[:, :, :, 3], tmp_res[:, :, :, 4]
            nmethods_fromfile = nerr.shape[0]
            avg_nerr_list_min_nerr[noise_cnt, :nmethods_fromfile] = nerr.min(axis=2).mean(axis=1)
            avg_freg_list[noise_cnt, :nmethods_fromfile, :, :] = freg
            avg_freg_list_avg[noise_cnt, :nmethods_fromfile, :] = freg.mean(axis=1)
            L0[nones != cardi] = np.inf
            minidx0 = np.argmin(L0, axis=2)  # opt
            idx_correct_cardi =(nones == cardi)
            if len(idx_correct_cardi) == 0:
                print('')
            L1[nones != cardi] = np.inf
            nerr_list_res = N//2+np.zeros((nerr.shape[0],nerr.shape[1]))
            for method_cnt in range(nerr.shape[0]):
                for avg_cnt in range(nerr.shape[1]):
                    if plot_mode == 0:
                        nerr_list_res[method_cnt, avg_cnt] = nerr[method_cnt, avg_cnt, minidx0[method_cnt, avg_cnt]]
                    else:
                        if np.sum(nones[method_cnt, avg_cnt, :] == cardi) > 0:
                            nerr_list_res[method_cnt, avg_cnt] = np.mean(nerr[method_cnt, avg_cnt, nones[method_cnt, avg_cnt, :] == cardi])
                        else:
                            nerr_list_res[method_cnt, avg_cnt] = np.mean(nerr[method_cnt, avg_cnt, :])

            avg_nerr_list_min_engy[noise_cnt, :nmethods_fromfile] = nerr_list_res.mean(axis=1)
    if cardi>=0:
        cardi_ = cardi
    else:
        cardi_ = N//2
    plot_name = 'test{:d}_avg{:d}_nic{:d}_nep{:d}K_cardi{:d}'.format(tid, plot_avg, plot_nic, nep // 1000, cardi_)

    plt.clf()
    plt.close()

    fig, (ax4) = plt.subplots(1, 1, sharex=False, sharey=False)

    if plot_mode==0:
        sel_str = 'NErr[Min(Loss)]'
        sel_str = 'Average Number of Errors'
    else:
        sel_str = 'Avg(NErr[Cardi])'
    plot_data=avg_nerr_list_min_engy
    for model_cnt in range(nmethods):
        if model_types_enable[model_cnt]:
            ax4.plot(noise_std_list, plot_data[:, model_cnt], plot_strings[model_cnt], label=method_names[model_cnt])
    if plot_ls:
        ax4.plot([0.05, 0.10, 0.15, 0.2], [0, 0, 0, 0.5], '-*', label='LightSolver')
    if opt_sol_en:
        ax4.plot(noise_std_list, avg_opt_nerr, '-gs', label='Optimal')
    ax4.set_xlabel(r'Noise STD $\sigma$')
    ax4.legend(frameon=False)
    ax4.grid()
    ax4.set_ylabel(sel_str)
    ax4.set_title('[{:d}x{:d}], Cardinality={:d}'.format(M,N,cardi_))
    fig.savefig('plots/'+plot_name+'_L0.pdf', format='pdf')
    fig.show()


if __name__ == '__main__':

    np.random.seed(444)
    set_start_method('spawn')

    time0 = time.time()

    create_folders(['data','datain','plots','tmp'])

    for tid in tests_run_list:
        avg_factor, nic, epochs, regtype, nbits = np.int32(testparams[tid][:5])
        assert (nbits > 0)
        lasso_reg_factor = testparams[tid][5]
        N, M = np.int32(testparams[tid][6:8])
        Cardi = np.int32(testparams[tid][9])
        force_rerun = np.int32(testparams[tid][10])
        force_regen = np.int32(testparams[tid][11])
        rmdiag = np.int32(testparams[tid][12])
        # if Cardi<0:
        #     Cardi = N//2
        print('test=', tid, ', noise=', noise_std_list, ', avg=', avg_factor, ', nic=', nic, ', epochs=', epochs, ', lr=', lr, ', reg=', lasso_reg_factor,', nbits=', nbits,', ', M, 'x', N, ', Cardi=', Cardi)
        name = 'test'+str(tid)+'_reg' + str(lasso_reg_factor) + '_avg' + str(avg_factor) + '_nic' + str(nic) + '_nep' + str(
            round(np.floor(epochs / 1000))) + 'k_Cardi'+str(int(Cardi))



        if run_mode == 0 or run_mode == 2:  # run tests
            run_experiment(noise_std_list, avg_factor, nic, epochs, lr, lasso_reg_factor, regtype, nbits, N, M,
                           Cardi, name, model_types_enable, tid, force_rerun, force_regen, use_multi_threading, stop_if_0_errors, opt_sol_en, rmdiag)
            print('noise=', noise_std_list, ', avg=', avg_factor, ', nic=', nic, ', epochs=', epochs, ', lr=', lr, ', reg=', lasso_reg_factor, ', ', M, 'x', N, ', Cardi=', Cardi)
            print('time:', round(time.time()-time0), 'sec')
        if run_mode == 1 or run_mode == 2:  # plot results
            plot_ls = (Cardi == 30 and N == 160)
            noise_std_list.sort()
            plot_avg, plot_nic = avg_factor, nic
            plot_results(tid, N, M, name, noise_std_list, plot_avg, plot_nic, epochs, Cardi, model_types_enable, method_names, show=True, plot_ls=plot_ls, plot_mode=plot_mode)
