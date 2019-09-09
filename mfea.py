from individual import Individual
from operators import crossover, mutate, DM
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random,math
def mfea(tasks,
         pop = 100,
         gen = 1000,
         TH = 0.9,
         selection_process = 'elitist',
         rmp = 0.7,
         mutation_rate=1,
         p_il = 0,
         reps = 20,
         method = 'L-BFGS-B',
         plot = False):

    '''
    :param tasks: List of Task type, can not be empty
    :param pop: Integer, population size
    :param gen: Integer, generation
    :param selection_process: String, only can be 'elitist' or 'roulette wheel', or can be customized
    :param rmp: Float, between 0 and 1
    :param p_il: Float, between 0 and 1
    :param reps: Integer, Repetition times
    :param method: String, details can be seen in document of scipy.optimize.minimize
    :param plot: Boolean, True or false
    :return: TotalEvaluations, ndarray, shape = (reps, gen)
              bestobj, ndarray, shape = (reps, gen, no_of_tasks))
              bestind, ndarray, shape = (shape=(reps, no_of_tasks, D_multitask))
    '''

    assert len(tasks) >= 1 and pop % 2 == 0
    if (pop % 2 != 0): pop += 1

    no_of_tasks = len(tasks)
    D = np.zeros(shape=no_of_tasks)
    for i in range(no_of_tasks):
        D[i] = tasks[i].dim
    D_multitask = int(np.max(D))

    fnceval_calls = np.zeros(shape=reps)
    TotalEvaluations = np.zeros(shape=(reps, gen))
    bestobj = np.empty(shape=(reps, gen, no_of_tasks))#result each gen
    bestind = np.empty(shape=(reps, no_of_tasks, D_multitask))#best result to save so far
    
    m=int(pop/no_of_tasks)#remain per task
    #calculate fm:
    a1=(TH-1)/(m-1)
    b1=(m-TH)/(m-1)
    a2=(0-TH)/(pop-m)
    b2=(pop*TH)/(pop-m)

    plotx=[]
    ploty1=[]
    ploty2=[]
    
    for rep in range(reps):
        et=0
        se=0
        f=np.random.uniform(0.3,0.7)
        print('Repetition: '+str(rep)+' :')

        population = np.asarray([Individual(D_multitask, tasks) for _ in range(2*pop)])
        factorial_costs = np.full(shape=(1 * pop, no_of_tasks), fill_value=np.inf)
        best_tmp = np.full(shape=no_of_tasks, fill_value=np.Inf)
        calls_per_individual = np.zeros(shape=pop)

        
        factorial_ranks = np.empty(shape=(1 * pop, no_of_tasks))
        tmp_factorial_ranks = np.empty(shape=(2 * pop, no_of_tasks))
        ability_vector = np.empty(shape=(1 * pop, no_of_tasks))
        for i, individual in enumerate(population[:pop]):
            for j in range(no_of_tasks):
                individual.skill_factor = j
                j, factorial_cost, calls_per_individual[i] = individual.evaluate(p_il, method)
                et+=1
                factorial_costs[i, j] = factorial_cost#i-th individual for j-th task
        for j in range(no_of_tasks):#get factorial rank
            factorial_cost_j = factorial_costs[:, j]
            indices = list(range(len(factorial_cost_j)))
            indices.sort(key=lambda x: factorial_cost_j[x])
            ranks = np.empty(shape=1 * pop)
            for i, x in enumerate(indices):
                ranks[x] = i + 1
            factorial_ranks[:, j ]= np.array(ranks)
        
        for j in range(no_of_tasks):
            bestind[rep,j,:D_multitask]=population[factorial_ranks[:,j].argmin()].rnvec[:D_multitask]

        for i in range(pop):#get ability vector
            for j in range(factorial_ranks.shape[1]):
                if factorial_ranks[i,j]<=m:
                    ability_vector[i,j]=a1*factorial_ranks[i,j]+b1
                else:
                    ability_vector[i,j]=a2*factorial_ranks[i,j]+b2
        
        
        
        fnceval_calls[rep] = fnceval_calls[rep] + np.sum(calls_per_individual)

        mu = 1
        mum = 39
        # generation = 0
        for generation in range(gen):
            new_factorial_cost=np.full(shape=(no_of_tasks,1 * m, no_of_tasks), fill_value=np.inf)
            kti=0
            for j in range(no_of_tasks):#for task 1 to k
                dim=tasks[j].dim
                group_j=[]
                for i in range(pop):
                    if factorial_ranks[i,j]<=m:
                        group_j.append(i)#get group j
                
                offspring_group_j=np.asarray([Individual(D_multitask, tasks) for _ in range(m)])
                offspring_ability_vector=np.empty(shape=(m, no_of_tasks))
                count=0
                while(count<m-1):
                    pa1,pa2=random.sample(group_j,2)
                    #print(pa1,pa2)
                    p1=population[pa1]
                    p2=population[pa2]
                    c1=offspring_group_j[count]
                    c2=offspring_group_j[count+1]
                    
                    if np.random.uniform()<rmp:
                        u = np.random.uniform(size=D_multitask)
                        cf = np.empty(shape=D_multitask)
                        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (mu + 1)))
                        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (mu + 1)))
    
                        c1.rnvec = crossover(p1.rnvec, p2.rnvec, cf)
                        c2.rnvec = crossover(p2.rnvec, p1.rnvec, cf)

                        if np.random.uniform()<mutation_rate:
                            c1.rnvec = DM(c1.rnvec,p1.rnvec,p2.rnvec,bestind[rep,j,:D_multitask],f)
                            c2.rnvec = DM(c2.rnvec,p2.rnvec,p1.rnvec,bestind[rep,j,:D_multitask],f)
                    else:
                        c1.rnvec = mutate(p1.rnvec, D_multitask, mum)
                        c2.rnvec = mutate(p2.rnvec, D_multitask, mum)
                    #inherit ability vector
                    if np.random.uniform()>0.5:
                        offspring_ability_vector[count]=np.array(ability_vector[pa1,:])
                        offspring_ability_vector[count+1]=np.array(ability_vector[pa2,:])
                    else:
                        offspring_ability_vector[count]=np.array(ability_vector[pa2,:])
                        offspring_ability_vector[count+1]=np.array(ability_vector[pa1,:])
                    count+=2
                #offspring generate over
                tmp_factorial_cost=np.full(shape=(1 * m, no_of_tasks), fill_value=np.inf)
                
                
                for i, abv in enumerate(offspring_ability_vector):
                    tasks_factorial_cost=np.zeros(shape=no_of_tasks)
                    for ts in range(no_of_tasks):
                        if ts==j or random.random()<abv[ts]:
                            offspring_group_j[i].skill_factor=ts
                            tss, factorial_cost, tmp= offspring_group_j[i].evaluate(p_il, method)
                            et+=1
                            
                            if ts!=j:
                                kti+=1
                                se+=1
                        else:
                            factorial_cost=np.inf
                        tasks_factorial_cost[ts]=factorial_cost
                    tmp_factorial_cost[i]=np.array(tasks_factorial_cost)
                new_factorial_cost[j]=np.copy(tmp_factorial_cost)
                
                if j==0:
                    intermediate_pop=np.concatenate((population[:pop],offspring_group_j),axis=0)
                else:
                    intermediate_pop=np.concatenate((intermediate_pop,offspring_group_j),axis=0)
            #end for loop:tasks
            
            #concatenate pop and all offspirngs
            #update pop by selecting factorial_cost:
            for j in range(no_of_tasks):
                if j==0:
                    intermediate_fac=np.concatenate((factorial_costs[:pop],new_factorial_cost[j]),axis=0)
                else:
                    intermediate_fac=np.concatenate((intermediate_fac,new_factorial_cost[j]),axis=0)
            for j in range(no_of_tasks):#get "factorial" rank
                factorial_cost_j = intermediate_fac[:, j]
                indices = list(range(len(factorial_cost_j)))
                indices.sort(key=lambda x: factorial_cost_j[x])
                ranks = np.empty(shape=2 * pop)
                for i, x in enumerate(indices):
                    ranks[x] = i + 1
                tmp_factorial_ranks[:, j ]= np.array(ranks)        
            count=0     
            for j in range(no_of_tasks):
                for i,indi in enumerate(intermediate_pop):
                    if tmp_factorial_ranks[i,j]<=m:
                        population[count].rnvec=np.copy(indi.rnvec)
                        factorial_costs[count]=np.array(intermediate_fac[i])
                        count+=1
            #update ability vector
            for j in range(no_of_tasks):#get factorial rank
                factorial_cost_j = factorial_costs[:, j]
                indices = list(range(len(factorial_cost_j)))
                indices.sort(key=lambda x: factorial_cost_j[x])
                ranks = np.empty(shape=1 * pop)
                for i, x in enumerate(indices):
                    ranks[x] = i + 1
                factorial_ranks[:, j ]= np.array(ranks)
            
            for i in range(pop):#get ability vector
                for j in range(factorial_ranks.shape[1]):
                    if factorial_ranks[i,j]<=m:
                        ability_vector[i,j]=a1*factorial_ranks[i,j]+b1
                    else:
                        ability_vector[i,j]=a2*factorial_ranks[i,j]+b2
            #update bestind
         
            
            for j in range(no_of_tasks):
                xxx = np.argmin(factorial_costs[:,j])
                if(best_tmp[j] > factorial_costs[xxx, j]):
                    bestobj[rep, generation, j] = factorial_costs[xxx, j]
                    best_tmp[j] = factorial_costs[xxx, j]
                    bestind[rep,j,:D_multitask]=population[xxx].rnvec[:D_multitask]

            if generation % 100 ==0 and True:#show result every gen
                print('Generation '+str(generation)+' :')
                print('Best objective of tasks : ',end='')
                print(best_tmp) 
                
                
            if (generation%10==0 or generation==gen-1) and True:#show process
                process=int((generation/(gen-1))*100)
                print('\r%{:3}['.format(process)+'*'*(process//5)+'->'+'.'*(20-process//5)+']',end='')
            
            if plot == True and generation%5==0:
                plotx.append(generation)
                ploty1.append(np.corrcoef(factorial_ranks[:,0],factorial_ranks[:,1])[0,1])
                ploty2.append(((kti/pop)-0.5)*2)
#                ploty1.append(math.log10(best_tmp[0]))
#                ploty2.append(math.log10(best_tmp[1]))
               
            

        print('Generation ' + str(generation) + ' :')
        print('Best objective of tasks : ', end='')
        print(best_tmp,'e times:',et,'more e times:',se,'f:',f)
        
        if plot == True:
            legend=['localRs','KTI']
            plt.plot(plotx,ploty1,'-b')
            plt.plot(plotx,ploty2,'-r')
            plt.legend(legend)
            plt.show()

        for j in range(no_of_tasks):
            dim = tasks[j].dim
            nnn = np.argmin(factorial_costs[:,j])
            bestobj[rep, generation, j] = factorial_costs[nnn, j]
            bestind[rep, j, :dim] = population[nnn].rnvec[:dim]


    

    return TotalEvaluations, bestobj, bestind