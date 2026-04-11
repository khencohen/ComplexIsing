import os
from qubo_network_def import ComplexNet
import numpy as np

def gen_random_ic(tid, noise_std, avg_cnt, ic_cnt, force_regen, N, save_to_file=1):
    ww_ic_fname = 'dataIn/test' + str(tid) + '_noise' + str(noise_std) + '_avg' + str(avg_cnt)+ '_ic' + str(ic_cnt) + '.txt'
    if os.path.exists(ww_ic_fname) and not force_regen:
        ww_random_ic = np.loadtxt(ww_ic_fname)
    else:
        ww_random_ic = 2*np.random.rand(N, 4)-1
        if save_to_file:
            np.savetxt(ww_ic_fname, ww_random_ic)
    return ww_random_ic

def training_task(tid, model_types_en, noise_std, nic, avg_cnt, A, m, e, b, nbits, Cardi, epochs, lr, lasso_reg_factor, stop_if_no_err_en, descript_en=True):
    print('avg_cnt=', avg_cnt)
    M, N = A.shape[0], A.shape[1]
    if M < N:
        f_reg_alpha = 0.001
    else:
        f_reg_alpha = 0.0

    upd_mask = np.array(model_types_en) == 1
    nmethods = len(model_types_en)
    best_acc = np.zeros(nmethods)
    res_array = np.zeros((nmethods, nic, 5))
    f_reg = lasso_reg_factor.repeat(nmethods)

    models_list = []
    for model_type_cnt in range(len(model_types_en)):
        tmp_model = ComplexNet(A, m, e, b, nbits, model_type=model_type_cnt, nep_switch_to_PB=epochs // 2, tb_enable=0, avg_cnt=avg_cnt, noise_std=noise_std)
        models_list.append(tmp_model)

    for ic_cnt in range(nic):
        ww_random = gen_random_ic(tid, noise_std, avg_cnt, ic_cnt, 0, N)
        if descript_en:
            print('\ntest={:d} noise={:.3f} reg={:.3f} avg={:d} nic={:d}/{:d}'
              .format(tid, noise_std, lasso_reg_factor, avg_cnt + 1, ic_cnt + 1, nic))
            print('f_reg=', f_reg)

        for model_type_cnt in range(len(model_types_en)):
            if model_types_en[model_type_cnt] == 1:
                if best_acc[model_type_cnt] < 1.0 or not stop_if_no_err_en:
                    mymodel = models_list[model_type_cnt]
                    mymodel.set_weights(ww_random)
                    train_res = mymodel.train(epochs, lr=lr, lasso_reg_factor=f_reg[model_type_cnt], show=False, descript_enable=descript_en, noise_std=noise_std, avg=avg_cnt, ic=ic_cnt)
                    accuracy_res, num_nonzeros_res, num_errors_res, opt_loss, fin_loss = train_res[0], train_res[1], train_res[2], train_res[3], train_res[4]

                    if descript_en:
                        print('\nnoise={:.3f}, avg_cnt={:d}, nic={:d}/{:d}, model={:d}, #ones={:d}, #errors={:d} , accuracy={:.4f}, opt_loss={:f}, fin_loss={:f}'.
                            format(noise_std, avg_cnt, ic_cnt + 1, nic, model_type_cnt, num_nonzeros_res, num_errors_res, accuracy_res, opt_loss, fin_loss))
                    res_array[model_type_cnt, ic_cnt, 0] = num_errors_res
                    res_array[model_type_cnt, ic_cnt, 1] = num_nonzeros_res
                    res_array[model_type_cnt, ic_cnt, 2] = f_reg[model_type_cnt]
                    res_array[model_type_cnt, ic_cnt, 3] = opt_loss
                    res_array[model_type_cnt, ic_cnt, 4] = fin_loss
                    if accuracy_res > best_acc[model_type_cnt]:
                        best_acc[model_type_cnt] = accuracy_res
                        res_array[model_type_cnt, ic_cnt:nic, 2] = f_reg[model_type_cnt]
                    f_reg[model_type_cnt] = f_reg[model_type_cnt] + f_reg_alpha*(num_nonzeros_res-Cardi)
        if (not any(best_acc[upd_mask] < 1.0)) and stop_if_no_err_en:
            break
    np.save('tmp/'+str(noise_std)+'_'+str(avg_cnt), res_array)
    return

