import genetic_algorithm as ga
import copy as cp

def TimeWindowGA(itertimes,popsize,w,q):   #w为时间窗长度，q为步长
    i=0
    resN=[]
    resR=[]
    length=cp.copy(ga.Data.F)
    Parents=ga.Population(popsize)
    while i<=length:
        temp=ga.algorithm_iter(popsize,itertimes,[i,i+w])#
        resN.append(temp.NXP[0:q])
        resR.append(temp.runway[0:q])
        i+=q
    best=ga.Individual(ga.Data)
    best.generate_NXP(resN)
    best.generate_runway(resR)
    best.generate_decision_vars()
    return best

if __name__=='__main':
    result=TimeWindowGA(100,200,2,3)
    print(result.cal_fitness())
    print("NXP:",result.NXP)