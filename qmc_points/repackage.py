from numpy import load, save

Nvals = [2**k for k in range(4, 12)]
Nvals.append(2**13)
for N in Nvals:
    filename = f'SPOD_N{N}_dim256'
    data = load(filename+'.npz')
    save(filename+'.npy', data['P'])
