import numpy as np

# Simulated Binary Crossover
def crossover(p1, p2, cf):
    # p1, p2 为genotype， ndarray类型, shape = (dim, )
    child = 0.5 * ((1+cf) * p1 + (1-cf) * p2)
    child[child < 0] = 0
    child[child > 1] = 1
    return child

# polynomial mutation
def mutate(p, dim, mum):
    # p 为genotype， ndarray类型
    # dim = p.shape[0]
    p_tmp = np.copy(p)
    for i in range(dim):
        u = np.random.uniform()
        if u < 0.5:
            delta = (2 * u) ** (1 / (1 + mum)) - 1
            p_tmp[i] = p[i] + delta * p[i]
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
            p_tmp[i] = p[i] + delta * (1 - p[i])
    p_tmp[p_tmp<0]=0
    p_tmp[p_tmp>1]=1
    return p_tmp
        
def DM(y1,y2,y3,y4,f):
#    f=0.5#np.random.uniform()
    result=y1+f*(y4-y1+y2-y3)
    result[result<0]=0
    result[result>1]=1
    return result
