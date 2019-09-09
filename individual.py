import numpy as np
import scipy.optimize as op

class Individual(object):

    def __init__(self, D_multitask, tasks):
        self.dim = D_multitask
        self.tasks = tasks
        self.no_of_tasks = len(tasks)

        self.rnvec = np.random.uniform(size=D_multitask)
        self.scalar_fitness = None
        self.skill_factor = None

    def evaluate(self, p_il,method):
        if self.skill_factor == None:
            raise ValueError("skill factor not set")
        else:
            task = self.tasks[self.skill_factor]
            nvars = self.rnvec[:task.dim]
            vars = task.decode(nvars)
            objective = task.fnc(vars)
        return self.skill_factor, objective, 1#func count
