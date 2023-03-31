import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

def reference(x): #la densité de référence à estimer
        return 1/(np.sqrt(2*np.pi))*np.exp(-x**2/2)



#Partie 1
#Question 1
def K1(x):
    
    
def K2(x):
        

def K3(x):
        

def K4(x):
     
#Question 2
def AllplotK(pas,xmin,xmax,col1,col2,col3,col4):
    


#Question 3
n=100
X=


#Question 4
def fchapeau(funct,h,x): #l'estimation de la densite f (ici la gaussienne standard, pour une fenetre h, au point x pour le noyau funct)
        

#Question 5        
def Allplotfchapeauh2(xmin,xmax,pas,col1,col2,col3,col4,colref):
        

#Question 6
def Allplotfchapeauh1(xmin,xmax,pas,col1,col2,col3,col4,colref):
    
#Question 7

#Question 8    
def SCE(funct,h,f):
    
#Question 9
def lemeilleurh(funct,f):
     
#Question 10 
def Allplotfchapeauhoptimal(xmin,xmax,pas,col1,col2,col3,col4,colref):
        


##Partie 2
def estimationdensite(N,h,mu1,sigma1,mu2,sigma2):
        # générer l'échantillon à partir de deux lois normales
        X = np.concatenate((np.random.normal(mu1, sigma1, int(0.3 * N)),
                            np.random.normal(mu2, sigma2, int(0.7 * N))))[:, np.newaxis]

        # préparer les points où on calculera la densité
        X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

        # préparation de l'affichage de la vraie densité, qui est celle à partir
        #  de laquelle les données ont été générées (voir plus haut)
        # la pondération des lois dans la somme est la pondération des lois
        #  dans l'échantillon généré (voir plus haut)
        true_density = (0.3 * norm(mu1,sigma1).pdf(X_plot[:,0]) + 0.7 * norm(mu2,sigma2).pdf(X_plot[:,0]))

        # estimation de densité par noyaux gaussiens
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)   


        # calcul de la densité pour les données de X_plot
        density = np.exp(kde.score_samples(X_plot))

        # affichage : vraie densité et estimation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
        ax.plot(X_plot[:,0], density, '-', label="Estimation")
        ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
        ax.legend(loc='upper left')
        plt.show()           


def estimationdensite2(N,h,mu1,sigma1,mu2,sigma2):
        # générer l'échantillon à partir de deux lois normales
        X = np.concatenate((np.random.normal(mu1, sigma1, int(0.3 * N)),
                            np.random.normal(mu2, sigma2, int(0.7 * N))))[:, np.newaxis]

        # préparer les points où on calculera la densité
        X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

        # préparation de l'affichage de la vraie densité, qui est celle à partir
        #  de laquelle les données ont été générées (voir plus haut)
        # la pondération des lois dans la somme est la pondération des lois
        #  dans l'échantillon généré (voir plus haut)
        true_density = (0.3 * norm(mu1,sigma1).pdf(X_plot[:,0]) + 0.7 * norm(mu2,sigma2).pdf(X_plot[:,0]))

        # estimation de densité par noyaux d'epanechnikov
        kde = KernelDensity(kernel='epanechnikov', bandwidth=h).fit(X)   


        # calcul de la densité pour les données de X_plot
        density = np.exp(kde.score_samples(X_plot))

        # affichage : vraie densité et estimation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill(X_plot[:,0], true_density, fc='b', alpha=0.2, label='Vraie densité')
        ax.plot(X_plot[:,0], density, '-', label="Estimation")
        ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
        ax.legend(loc='upper left')
        plt.show()      


        
        