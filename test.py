#%%
import re
import os
from numpy import percentile
from typing import Tuple
from functools import wraps

REPEAT = 100
PARAMETERS = None
DIR_TO_SAVE = 'run_results/'
ITERATE_LIST_CUDA = ['_MM','_MV','_VV']
ITERATE_LIST_CPU = ['_SEQ','_OMP']
TEST_CPU = False
Q = 0

def imap(typename, iterable):
    return [typename(it) for it in iterable]

def save(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        global PARAMETERS,Q,TEST_CPU
        iter_list = ITERATE_LIST_CPU if TEST_CPU else ITERATE_LIST_CUDA
        ilen = len(iter_list)
        fname = '_'.join(PARAMETERS)+iter_list[Q%ilen]+'.txt'
        Q += 1
        if not os.path.exists(DIR_TO_SAVE+fname):
            with open(DIR_TO_SAVE+fname,'w') as f:
                f.write(' '.join(map(str,args[0])))
        return function(*args, **kwargs)
    return wrapper

def tested(param, reduce):
    iter_list = ITERATE_LIST_CPU if TEST_CPU else ITERATE_LIST_CUDA
    fname = '_'.join(param)+iter_list[-1]+'.txt'
    if not os.path.exists(DIR_TO_SAVE+fname):
        return None
    result = []    
    for it in iter_list:
        fname = '_'.join(param)+it+'.txt'
        with open(DIR_TO_SAVE+fname) as f:
            r = f.readline()
            result.append(reduce(imap(float, r.split())))
    return result
                
@save
def box(data):
    med, boxTop, boxBottom, whiskerTop, whiskerBottom = list(percentile(data, [50, 75, 25, 95, 5]))
    # IQR = boxTop - boxBottom
    # whiskerTop = boxTop + 1.5*IQR
    # whiskerBottom = boxBottom - 1.5*IQR
    return med, boxTop, boxBottom, whiskerTop, whiskerBottom

# Test OpenMP
def test_omp(n_rows = 5000, n_cols = 5000, n_threads = 8, average_over = 1, reduce = box, show_result = False) -> Tuple[float]:
    global PARAMETERS
    assert n_rows != 0 and n_cols != 0
    sequential, parallel = [], []
    command = f'matrixAdd_omp.exe -r {n_rows} -c {n_cols} -t {n_threads} -a {average_over}'
    PARAMETERS = imap(str, [n_rows, n_cols, n_threads, average_over, REPEAT])
    result = tested(PARAMETERS, reduce)
    if result != None:
        print('tested.')
        return result
    print(command,end=' ...')
    for i in range(REPEAT):
        with os.popen(command) as pipe:
            result = pipe.read()
        pattern = r'Time Elapsed: (\d+\.\d+)\(ms\)'
        cpu_seq_time, cpu_omp_time = map(float, re.findall(pattern, result))
        sequential.append(cpu_seq_time)
        parallel.append(cpu_omp_time)
        if show_result:
            if i==0:
                print()
            print(f'SEQ = {cpu_seq_time:.3f} ms, OMP = {cpu_omp_time:.3f} ms')
    print('done')
    return reduce(sequential), reduce(parallel)

def test_cuda(n_rows = 5000, n_cols = 5000, blockDim1D = 1024, xBlockDim2D = 32, N = 1, reduce = box, show_result = False) -> Tuple[float]:
    global PARAMETERS
    assert n_rows != 0 and n_cols != 0
    yBlockDim2D = blockDim1D // xBlockDim2D
    mat_mat, mat_vec, vec_vec = [], [], []
    command = f'matrixAdd_cuda.exe -r {n_rows} -c {n_cols} -b {blockDim1D} -x {xBlockDim2D} -y {yBlockDim2D} -N {N}'
    PARAMETERS = imap(str, [n_rows, n_cols, blockDim1D, xBlockDim2D, N, REPEAT]) 
    result = tested(PARAMETERS, reduce)
    if result != None:
        print('tested.')
        return result
    print(command,end=' ...')
    for i in range(REPEAT):
        with os.popen(command) as pipe:
            result = pipe.read()
        pattern = r'Time Elapsed: (\d+\.\d+)\(ms\)'
        try:
            time_2dmat_2dmem, time_2dmat_1dmem, time_1dmat_1dmem = map(float, re.findall(pattern, result))
        except:
            print('\nError when parsing, original output:\n',result)
        mat_mat.append(time_2dmat_2dmem)
        mat_vec.append(time_2dmat_1dmem)
        vec_vec.append(time_1dmat_1dmem)
        if show_result:
            if i==0:
                print()
            print(f'{time_2dmat_2dmem:.3f}, {time_2dmat_1dmem:.3f}, {time_1dmat_1dmem:.3f}')
    print('done')
    return reduce(mat_mat), reduce(mat_vec), reduce(vec_vec)

#%%
mms, mvs, vvs = [], [], []
for n in range(3,7+1):
    mm, mv, vv = test_cuda(n_rows=n*1000)
    mms.append([n,*mm])
    mvs.append([n+.2,*mv])
    vvs.append([n+.4,*vv])

for tdim,mdim,data in [(1,1,vvs),(2,1,mvs),(2,2,mms)]:
    with open(f'n_rows{tdim}{mdim}.dat','w') as f:
        f.writelines('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
print('result saved.')
# %%
mms, mvs, vvs = [], [], []
for n in range(3,7+1):
    mm, mv, vv = test_cuda(n_cols=n*1000)
    mms.append([n,*mm])
    mvs.append([n+.2,*mv])
    vvs.append([n+.4,*vv])

for tdim,mdim,data in [(1,1,vvs),(2,1,mvs),(2,2,mms)]:
    with open(f'n_cols{tdim}{mdim}.dat','w') as f:
        f.writelines('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
print('result saved.')
#%%
mms, mvs, vvs = [], [], []
for n in range(4,10+1):
    print(2**n, 2**((n+1)//2), 2**n / 2**((n+1)//2))
    mm, mv, vv = test_cuda(blockDim1D=2**n, xBlockDim2D=2**((n+1)//2))
    mms.append([n,*mm])
    mvs.append([n+.2,*mv])
    vvs.append([n+.4,*vv])

for tdim,mdim,data in [(1,1,vvs),(2,1,mvs),(2,2,mms)]:
    with open(f'blockDim_{tdim}{mdim}.dat','w') as f:
        f.writelines('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
print('result saved.')
# %%
# num_elements_per_thread.
mms, mvs, vvs = [], [], []
for n in range(0,4+1):
    mm, mv, vv = test_cuda(5000, N=2**n)
    vvs.append([2**n,*vv])

for tdim,mdim,data in [(1,1,vvs)]:
    with open(f'N_{tdim}{mdim}.dat','w') as f:
        f.writelines('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
print('result saved.')
# %%
mms, mvs, vvs = [], [], []
for n in range(1,5+1):
    num_elements = 25000000
    n_rows = 10**n * 5
    mm, mv, vv = test_cuda(n_rows=n_rows, n_cols=num_elements//n_rows)
    mms.append([n,*mm])
    mvs.append([n+.2,*mv])
    vvs.append([n+.4,*vv])

for tdim,mdim,data in [(1,1,vvs),(2,1,mvs),(2,2,mms)]:
    with open(f'n_elems_{tdim}{mdim}.dat','w') as f:
        f.writelines('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
print('result saved.')

#%%
mms, mvs, vvs = [], [], []
for n in range(0,4+1):
    n_rows = 10**n * 5
    mm, mv, vv = test_cuda(n_rows=n_rows)
    mms.append([n,*mm])
    mvs.append([n+.2,*mv])
    vvs.append([n+.4,*vv])

for tdim,mdim,data in [(1,1,vvs),(2,1,mvs),(2,2,mms)]:
    with open(f'matRows_{tdim}{mdim}.dat','w') as f:
        f.write('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
print('result saved.')
# %%
TEST_CPU = True
Q = 0
#%%
seqs, omps = [], []
for n in range(0,4+1):
    n_rows = 10**n * 5
    seq, omp = test_omp(n_rows=n_rows, average_over=max(1,5000//n_rows))
    seqs.append([n,*seq])
    omps.append([n+.2,*omp])

for name, data in [('seq',seqs), ('omp',omps)]:
    with open(f'OMP_{name}.dat','w') as f:
        f.write('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
#%%
seqs, omps = [], []
for n in range(1,12+1):
    seq, omp = test_omp(n_threads=n)
    omps.append([n,*omp])

for name, data in [('omp',omps)]:
    with open(f'OMP_NThreads{name}.dat','w') as f:
        f.write('\n'.join(' '.join(map(str,map(lambda x: round(x,3), line))) for line in data))
# %%
