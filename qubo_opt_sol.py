import numpy as np

def find_opt_sol(A, b, C, v=0, non_zero_cnt=0, start_idx=0, er2_min=np.inf, v0=0):
    N = A.shape[1]
    if non_zero_cnt == 0:
        v = np.zeros((N, 1))
    non_zero_cnt += 1
    for n in range(start_idx, N):
        v[n] = 1
        if non_zero_cnt == C:
            assert (sum(v[:, 0] > 0) == C)
            err2 = np.square(abs(A@v-b)).sum()
            if err2 < er2_min:
                er2_min = err2
                v0 = v.copy()
        else:
            v0, er2_min = find_opt_sol(A, b, C, v, non_zero_cnt, start_idx=n+1, er2_min=er2_min, v0=v0)
        v[n] = 0
    return v0, er2_min

def find_opt_sol_UnknownCardinality(A, b):
    N = A.shape[1]
    er2_min = np.inf
    v0 = np.zeros((N, 1))
    for n in range(2**N):
        v = np.array([(n & (1<<k)) for k in range(N)]).reshape((N, 1))
        err2 = np.square(abs(A@v-b)).sum()
        if err2 < er2_min:
            er2_min = err2
            for k in range(N):
                v0[k] = v[k].copy()
    return v0, er2_min

