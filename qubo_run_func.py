from qubo_train_func import training_task
import os

import numpy as np
import time

from multiprocessing import Process, set_start_method
from qubo_network_def import Axb
from qubo_opt_sol import find_opt_sol, find_opt_sol_UnknownCardinality

def gen_problem_data(tid, noise_std, avg_cnt, force_regen, N, M, Cardi, nbits):
    A_fname = 'datain/test' + str(tid) + '_noise' + str(noise_std) + '_avg' + str(avg_cnt) + '_A.txt'
    m_fname = 'datain/test' + str(tid) + '_noise' + str(noise_std) + '_avg' + str(avg_cnt) + '_m.txt'
    e_fname = 'datain/test' + str(tid) + '_noise' + str(noise_std) + '_avg' + str(avg_cnt) + '_e.txt'
    b_fname = 'datain/test' + str(tid) + '_noise' + str(noise_std) + '_avg' + str(avg_cnt) + '_b.txt'
    if os.path.exists(A_fname) and not force_regen:
        A = np.loadtxt(A_fname)
        m = np.loadtxt(m_fname).reshape((-1, 1))
        e = np.loadtxt(e_fname).reshape((-1, 1))
        b = np.loadtxt(b_fname).reshape((-1, 1))
    else:
        if Cardi>=0:
            cardi_prm = Cardi
        else:
            cardi_prm = N//2
        Axb_obj = Axb(N=N, M=M, Cardi=cardi_prm, nbits=nbits)
        A = Axb_obj.A
        m = Axb_obj.get_m(nbits)
        e = Axb_obj.get_e(noise_std)
        b = Axb_obj.get_b(m, e)
        np.savetxt(A_fname, A)
        np.savetxt(m_fname, m, fmt='%d')
        np.savetxt(e_fname, e)
        np.savetxt(b_fname, b)


    return A, m, e, b


def call_opt_sol_func(N, A, b, m, Cardi, avg_cnt, avg_factor, noise_std, opt_array_avg, total_opt_runtime):
    assert (N <= 32)
    opt_t0 = time.time()
    if Cardi>=0:
        v0, er2_min = find_opt_sol(A, b, Cardi)
    else:
        v0, er2_min = find_opt_sol_UnknownCardinality(A, b)
    total_opt_runtime += (time.time() - opt_t0)
    assert (sum(v0[:, 0] > 0) == Cardi)
    # print(noise_std, '{:d}/{:d}'.format(avg_cnt, avg_factor), v0[:, 0].T, er2_min)
    opt_nerr = sum(v0[:, 0] != m[:, 0])
    opt_array_avg[avg_cnt] = opt_nerr
    pass
    return total_opt_runtime


def run_experiment(noise_std_list, avg_factor, nic, epochs, lr, lasso_reg_factor, regtype, nbits, N, M, Cardi, name,
                   model_types_enable, tid, force_rerun, force_regen, use_multi_threading, stop_if_no_err_en, opt_sol_en, rmdiag):
    upd_mask = np.array(model_types_enable) == 1
    nmethods = len(model_types_enable)
    testdir = './data/test'+str(tid)
    if not os.path.isdir(testdir):
        os.mkdir(testdir)
    noise_cnt = -1
    opt_runtime = 0


    for noise_std in noise_std_list:
        print('Noise std:', noise_std)
        noise_cnt += 1
        res_current_noise_point_fname = 'data/test{:d}/res_{:s}_noise{:.3f}.npy'.format(tid,name,noise_std)
        opt_current_noise_point_fname = 'data/test{:d}/opt_noise{:.3f}.npy'.format(tid,noise_std)
        res_array_avg = np.zeros((nmethods, avg_factor, nic, 5)) + np.nan
        opt_array_avg = np.zeros((avg_factor, 1)) + np.nan
        if os.path.exists(res_current_noise_point_fname):
            res_array_avg_fromfile = np.load(res_current_noise_point_fname)
            assign_num = min(res_array_avg.shape[0], res_array_avg_fromfile.shape[0])
            res_array_avg[:assign_num, :, :, :] = res_array_avg_fromfile[:assign_num, :, :, :]
            run_this_point = force_rerun
        else:
            run_this_point = 1
        if run_this_point:
            n_proc = 6
            avg_tests_to_run = avg_factor
            num_of_runs = int(np.ceil(avg_factor / n_proc))

            for run_cnt in range(num_of_runs):
                tasks = []
                num_proc_to_run = min(n_proc, avg_tests_to_run)
                avg_tests_to_run -= num_proc_to_run

                for pid in range(num_proc_to_run-1):
                    avg_cnt = n_proc * run_cnt + pid
                    A, m, e, b = gen_problem_data(tid, noise_std, avg_cnt, force_regen, N, M, Cardi, nbits)
                    if opt_sol_en:
                        opt_runtime = call_opt_sol_func(N, A, b, m, Cardi, avg_cnt, avg_factor, noise_std, opt_array_avg, opt_runtime)

                    if sum(model_types_enable) > 0:
                        if use_multi_threading:
                            tasks.append(Process(target=training_task,
                                                 args=(tid, model_types_enable, noise_std, nic, avg_cnt, A, m, e, b, nbits,
                                                       Cardi, epochs, lr, lasso_reg_factor, stop_if_no_err_en)))
                        else:
                            training_task(tid, model_types_enable, noise_std, nic, avg_cnt, A, m, e, b, nbits,
                                    Cardi, epochs, lr, lasso_reg_factor, stop_if_no_err_en)
                if sum(model_types_enable) > 0:
                    if use_multi_threading:
                        print('#ExtraProcesses={:d}'.format(len(tasks)))
                        pass
                        for thread in tasks:
                            thread.start()
                    avg_cnt = n_proc * run_cnt + (num_proc_to_run-1)
                    A, m, e, b = gen_problem_data(tid, noise_std, avg_cnt, force_regen, N, M, Cardi, nbits)
                    if opt_sol_en:
                        opt_runtime = call_opt_sol_func(N, A, b, m, Cardi, avg_cnt, avg_factor, noise_std, opt_array_avg, opt_runtime)
                    print('Starting main process, avg_cnt=', avg_cnt)
                    training_task(tid, model_types_enable, noise_std, nic, avg_cnt, A, m, e, b, nbits,
                                  Cardi, epochs, lr, lasso_reg_factor, stop_if_no_err_en)

                    if use_multi_threading:
                        # Join the threads
                        for thread in tasks:
                            thread.join()
            if sum(model_types_enable) > 0:
                for avg_cnt in range(avg_factor):
                    res_array = np.load('tmp/' + str(noise_std) + '_' + str(avg_cnt) + '.npy')
                    res_array_avg[upd_mask, avg_cnt, :, :] = res_array[upd_mask, :, :]
            np.save(res_current_noise_point_fname, res_array_avg)
            np.save(opt_current_noise_point_fname, opt_array_avg)
            if opt_sol_en:
                np.save('data/test'+str(tid)+'/runtime_test'+str(tid), int(opt_runtime))
