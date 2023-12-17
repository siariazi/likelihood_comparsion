#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:05:05 2023

@author: siavashriazi
"""

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import copy

# the following function returns the ODE's
def sir_odes(pars,state):
    beta=pars[0]
    gamma=pars[1]
    psi=pars[2]
    sigma=pars[3]
    kappa=pars[4]
    
    
    t=state[0]
    S=state[1]
    I=state[2]
    R=state[3]
    
    
    dSdt = -beta/kappa * S * I+sigma*R
    dIdt = beta/kappa  * S * I -  (gamma + psi)* I
    dRdt = (gamma + psi) * I -sigma*R
    
    return np.array([1,dSdt,dIdt,dRdt])

def sir_rk4(pars,inits,nStep):
    h=pars[len(pars)-1]/nStep
    out=np.array([0.]+inits)
    temp=out
    for s in range(nStep):
        k1=sir_odes(pars,temp)
        fk1=temp+k1*h
        k2=sir_odes(pars,fk1)
        fk2=temp+k2*h/2
        k3=sir_odes(pars,fk2)
        fk3=temp+k3*h/2
        k4=sir_odes(pars,fk3)
        fk4=temp+k4*h
        temp=temp+(k1+2*k2+2*k3+k4)/6*h
        out=np.vstack((out,temp))
    return out

# stochastic model

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


class simTree:
    def __init__(self,pars):
        #setting parameters
        self.beta=pars[0]
        self.gamma=pars[1]
        self.psi=pars[2]
        self.sigma = pars[3]
        self.kappa = pars[4]
        self.i0 = pars[5]
        self.T=pars[6]
        
        #setting state
        self.treeMtrx = np.zeros((self.i0, self.i0))

        self.state = [1] * self.i0
        self.alive = [1] * self.i0
        self.epiState=np.array([[0,self.kappa-self.i0,self.i0,0]])
        #self.gillespie()#simulate tree
    def event(self,e,Deltat):
        #Add delta t to infected lineages
        self.treeMtrx=np.identity(len(self.treeMtrx))*Deltat*self.alive+self.treeMtrx
        #print(self.treeMtrx)
        if e==1: #infection
            ind=rnd.choice(find_indices(self.state, lambda x: x==1)) #pick parent
            #update tree matrix, state vector, alive vector
            self.treeMtrx=np.vstack((self.treeMtrx,self.treeMtrx[ind])) #add row to tree mtrx
            col=np.transpose(np.hstack((self.treeMtrx[ind],self.treeMtrx[ind,ind])))
            self.treeMtrx=np.vstack((np.transpose(self.treeMtrx),col))#adding column
            #print(self.treeMtrx)
            self.state=self.state+[1]
            self.alive=self.alive+[1]
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,-1,1,0]))
        elif e==2:#recovery
            ind=rnd.choice(find_indices(self.state, lambda x: x==1))# pick lineage to die
            self.state[ind]=0
            self.alive[ind]=0
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,-1,1]))
        elif e==3:#samplint
            ind=rnd.choice(find_indices(self.state, lambda x: x==1))# pick lineage for sampling
            self.state[ind]=-1
            self.alive[ind]=0
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,-1,1]))
        elif e==4:#waning
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,1,0,-1]))
        elif e==0: #Update to present day *empty event*
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,0,0]))
        else:
            print("ERROR in event id")
    def gillespie(self):
        #initialize
        t=0
        S=self.epiState[-1,1]
        I=self.epiState[-1,2]
        R=self.epiState[-1,3]
        rates=[self.beta/self.kappa*S*I,self.gamma*I,self.psi*I,self.sigma*R]
        totalRate = sum(rates)
        Deltat=round(np.random.exponential(scale=1/totalRate),3)
        e=rnd.choices(np.linspace(1,len(rates),len(rates)), weights=rates)[0]
        while t+Deltat<self.T:
            #perform event
            self.event(e,Deltat)
            t+=Deltat
            #pick new deltat
            S=self.epiState[-1,1]
            I=self.epiState[-1,2]
            R=self.epiState[-1,3]
            rates=[self.beta/self.kappa*S*I,self.gamma*I,self.psi*I,self.sigma*R]
            totalRate = sum(rates)
            if totalRate==0:
                Deltat=self.T-t
                e=0
            else:
                Deltat=round(np.random.exponential(scale=1/totalRate),3)
                e=rnd.choices(np.linspace(1,len(rates),len(rates)), weights=rates)[0]
        #Last step
        self.event(0,self.T-t)
        self.sampledTree()
    def sampledTree(self):
        # Extracts the sampled tree
        # Extracts the observed sampling times recoded FORWARD in time(yVec)
        # Extracts the observed birth times recoded FORWARD in time (xVec)
        inds=find_indices(self.state, lambda x: x==-1)
        self.sampTree=self.treeMtrx[inds][:,inds]
        self.yVec=np.diagonal(self.sampTree)# sampling times are the diagonal
        # birth times are the (non-duplicated) off diagonals greater than 0
        temp2=np.reshape(np.triu(self.sampTree, k=1),len(self.sampTree)*len(self.sampTree))
        temp2=[x for x in temp2 if x > 0]
        self.xVec=np.array(list(dict.fromkeys(temp2)))
    # a recursive matrix to break the matrix 
    def convert_newick(self,mat):
        if np.shape(mat)[0] == 1:
            #return(":"+str(mat[0][0]))
            return "xAz:" + str(mat[0][0])
        elif np.shape(mat)[0] == 2:
            new_mat = mat - np.amin(mat)
            # dv collects non zero elements of the new mat 
            dv = new_mat[np.nonzero(new_mat)]
            #return("(:"+str(dv[0])+",:"+str(dv[1])+"):"+str(np.amin(mat)))
            return "(xAz:" + str(dv[0]) + ",xAz:" + str(dv[1]) + "):" + str(np.amin(mat))
        elif np.shape(mat)[0] > 2:
            branch_length =  np.amin(mat)
            # substracting min value of all elements
            newm = mat - branch_length
            out = self.break_matrix(newm)
            return "(" + self.convert_newick(out[0])  + "," + self.convert_newick(out[1]) + "):" + str(branch_length)

    # break matrix breaks the matrix to two matrices
    def break_matrix(self,mat):
        mat2 = copy.deepcopy(mat)
        k = []
        for i in range(np.shape(mat2)[0]):
            if mat2[0][i] == 0:
                k.append(i)
            #print(i)
        m1 = np.delete(mat2,k,1)
        m1 = np.delete(m1,k,0)
        m2 = mat[np.ix_(k,k)]
        output = [m1,m2]
        return output

    # toNweick outputs the final result
    def toNewick(self):
        out = self.convert_newick(self.treeMtrx)
        self.treeTxt = "("+out+")xA0z;"
        #self.treeTxt = "("+out+");"
    
    def add_label(self):
        j = 1
        textl = list(self.treeTxt)
        label_list = []
        for i in range(0,len(textl)):
            #print(i)
            if textl[i] == 'A':
                textl.insert(i+1,j)
                label_list.append("A"+str(j))
                j += 1
                
        label_list.append("A0")
        self.treeTxtL = ''.join(map(str, textl))


class like():
    def __init__(self,parsL,xVec,yVec):
        #Parameters
        ##Epi parameters
        self.beta = parsL[0]
        self.gamma = parsL[1]
        self.psi = parsL[2]
        self.sigma = parsL[3]
        self.kappa = parsL[4]
        self.i0 = parsL[5]
        self.T = parsL[6]
        self.SIR=sir_rk4(parsL,[self.kappa-self.i0,self.i0,0],self.T*3)

        ##Diversification parameters for pylikelihood
        self.newTau = self.SIR[:,0] # a vector of time to be attached to E vector for plotting 
        self.inits = [1] # initial condition of E
        self.sValue = interp1d(self.SIR[:,0], self.SIR[:,1], kind = 'cubic')
        eVec = np.vstack((self.newTau,np.squeeze(self.E_euler(self.newTau))))
        self.eValue = interp1d(eVec[0,:], eVec[1,:], kind = 'cubic')
        intR = np.linspace(start=0.1, stop=self.T, num=100)
        self.f = interp1d(intR, list(map(self.phi,intR)), kind = 'cubic')
        
        # case likelihood
        self.iValue = interp1d(self.SIR[:,0], self.SIR[:,2], kind = 'cubic')
        self.sampValue = lambda t: self.iValue(t)*self.psi
        
        
        self.xVec2 = xVec
        self.yVec2 = yVec
        self.xVec = self.T- xVec
        self.yVec = self.T - yVec
        
        
    # calcE() calculates dEdt
    def calcE(self,state,tau):
        
        E=state[0]
        dEdtau = -(self.lamda(tau) + self.gamma + self.psi)*E + self.lamda(tau)*E**2 + self.gamma  
        return dEdtau
    
    def E_euler(self,time):
        inits = [1] # initial condition of E

        h = time[1] - time[0]
        out=np.array(inits)
        temp=out
        for i in time[1:]:
            dEdtau = self.calcE(temp,i)   
            temp=temp+dEdtau*h
            out=np.vstack((out,temp))
        return out
           
    def lamda(self,tau):
        lamda = self.sValue(T-tau)*self.beta/self.kappa
        #print("lamda is:",lamda)
        return lamda
    # phiA returns phi integrad
    # this function should have a return value because we are integrating it over integration interval in quad()   
    def phiA(self,tau):
        phiA =  2*self.lamda(tau)*self.eValue(tau) - (self.lamda(tau) + self.gamma + self.psi)
        return phiA
    
    # calculating phi (integral of phiA)
    def phi(self,tau):
        return quad(self.phiA, 0, tau)[0]
    
    # likelihood() calls other functions in the class and calculates the (log)likelihood
    def phyLL(self):
        self.phyL = self.f(self.T) + sum([np.log(self.lamda(x)) + self.f(x) for x in self.xVec]) + sum([np.log(psi) - self.f(y) for y in self.yVec])

    def cLL(self):
        #matching_rows = epiState[np.isin(epiState[:, 0], yVec)]
        #ivec = matching_rows[:, 2]
        
        self.cL = -1*quad(self.sampValue, 0, self.yVec2[0])[0] + np.log(self.sampValue(self.yVec2[0]))
        for t in range(1,len(self.yVec2)):
            self.cL += -1*quad(self.sampValue, self.yVec2[t-1], self.yVec2[t])[0] + np.log(self.sampValue(self.yVec2[t]))
        


#testPars=[0.25,0.05,0.01,0.001,300,3,100]#beta,gamma,psi,sigma,kappa,i0,T according to Mathematica
testPars=[0.12,0.05,0.01,0.001,300,3,100]#beta,gamma,psi,sigma,kappa,i0,T according to Mathematica


beta, gamma, psi, sigma, kappa, i0, T = testPars
testInits=[kappa-i0,i0,0]

testInits=[kappa-i0,i0,0]
SIR=sir_rk4(testPars,testInits,30)
#plt.plot(nInt[:,0],nInt[:,1],nInt[:,0],nInt[:,2],nInt[:,0],nInt[:,3])
# plotting ODE solution
sPlot, = plt.plot(SIR[:,0],SIR[:,1], label="Suscepible")
iPlot, = plt.plot(SIR[:,0],SIR[:,2], label="Infected")
rPlot, = plt.plot(SIR[:,0],SIR[:,3], label="Recovered")
plt.legend(handles=[sPlot,iPlot, rPlot])
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.show()

tree1=simTree(testPars)
tree1.gillespie()
np.shape(tree1.sampTree)

# plotting stochastic simulation
sPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,1], label="Suscepible")
iPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,2], label="Infected")
rPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,3], label="Recovered")
plt.legend(handles=[sPlot,iPlot, rPlot])
#plt.plot(tree1.epiState[:,0],tree1.epiState[:,1],tree1.epiState[:,0],tree1.epiState[:,2],tree1.epiState[:,0],tree1.epiState[:,3])
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.show()

np.shape(tree1.sampTree)

def LLcurve():
    global testPars
    gendat = [[],[],[]]

    for i in np.arange(0.01,0.5,0.01): # i is beta 
    #for i in np.arange(0.01,1,0.01): # i is gamma 

        gendat[0].append(i)
        pars=[i,testPars[1],testPars[2],testPars[3],testPars[4],testPars[5],testPars[6]] 
        #pars=[testPars[0],i,testPars[2],testPars[3],testPars[4],testPars[5],testPars[6]] 
        #print("pars is",pars)
        like1 = like(pars,tree1.xVec,tree1.yVec)
        like1.phyLL()
        gendat[1].append(like1.phyL)
        like1.cLL()
        gendat[2].append(like1.cL)
            
    #print("true value is:",testPars[0])
    gendata = np.array(gendat)
    plt.plot(gendata[0], gendata[1],color='b')  # blue is phylogenetic likelihood
    plt.plot(gendata[0], gendata[2],color='m')  # magneta is case likelihood
    plt.axvline(x = testPars[0], color = 'b', label = 'true value')
    plt.xlabel('beta')
    plt.ylabel('loglike')
    
LLcurve()
