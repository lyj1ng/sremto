
from mfea import mfea
from task import Task
from benchmark import CI_HS, CI_MS, CI_LS, PI_HS, PI_MS, PI_LS, NI_HS, NI_MS, NI_LS
from toyfnc import ackley, sphere, rastrigin
import numpy as np
if __name__=="__main__":
    print('test')
    
    repeat=1
    generation=851
    tasks = [Task(10, ackley, 50, -50),
             Task(10, sphere, 50, -50)]
    #
    #TotalEvaluations, bestobj, bestind = mfea(tasks,reps=1)
    ###print('TotalEvaluations:',TotalEvaluations)
    ###print('bestobj:',bestobj)
    ###print('bestind:',bestind)
#    tasks = CI_HS()
#    tasks = PI_LS()
    TotalEvaluations, bestobj, bestind = mfea(tasks,TH=0.3,gen=generation, reps=repeat,plot=True)
    result=np.zeros(shape=2)
    for rep in range(repeat):
        result+=bestobj[rep, generation-1]
    result/=repeat
    print('avg result:',result)
