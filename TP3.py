import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def reference(x):  # la densité de référence à estimer
    return 1/(np.sqrt(2*np.pi))*np.exp(-x**2/2)

# Partie 1
# Question 1


def K1(x):
    # le noyau uniforme
    return np.where(np.abs(x) <= 1, 1/2, 0)	# np.where(condition, si vrai, si faux)


def K2(x):
    # le noyau triangle
    return np.where(np.abs(x) <= 1, 1-np.abs(x), 0)


def K3(x):
    # le noyau epanechnikov
    return np.where(np.abs(x) <= 1, 3/4*(1-x**2), 0)


def K4(x):
    # le noyau gaussien
    return 1/(np.sqrt(2*np.pi))*np.exp(-x**2/2)


# Question 2
def AllplotK(pas, xmin, xmax, col1, col2, col3, col4):
    # affichage des noyaux
    x = np.arange(xmin, xmax, pas)
    plt.plot(x, K1(x), color=col1)
    plt.plot(x, K2(x), color=col2)
    plt.plot(x, K3(x), color=col3)
    plt.plot(x, K4(x), color=col4)
    # save figure to file
    plt.savefig('P1-BRETAGNOLLESPELOUTIERQ2.png', dpi=300)


pas = 0.01
xmin = -3
xmax = 3
col1 = 'red'
col2 = 'blue'
col3 = 'green'
col4 = 'orange'

AllplotK(pas, xmin, xmax, col1, col2, col3, col4)


# Question 3
# Générer une réalisation de l’échantillon aléatoire X selon la loi gaussienne standard de taille n. (n
# est pour l’instant fixé à 100 dans le script)
n = 100
X = np.random.normal(0, 1, n)
Xbis = np.random.randn(n)

# Vérification des 2 méthodes
print(X)
print(len(X))
print(Xbis)
print(len(Xbis))

# Les deux méthodes donnent bien le même résultat

# Question 4


def fchapeau(funct, h, x):  # l'estimation de la densite f (ici la gaussienne standard, pour une fenetre h, au point x pour le noyau funct)
    n = len(X)
    return 1/(n*h)*np.sum(funct((x-X)/h))


h = 0.5
x = 0

print(fchapeau(K1, h, x))

# Question 5
colref = 'black'
j=5 #question
g='_1' #numéro graphique
def Allplotfchapeauh2(xmin, xmax, pas, col1, col2, col3, col4, colref,h):
    x = np.arange(xmin, xmax, pas)
    fig, ax = plt.subplots()
    noms = ['Noyau uniforme', 'Noyau triangle', 'Noyau epanechnikov',
            'Noyau gaussien', 'Densité de référence']
    for i, K in enumerate([K1, K2, K3, K4]):
        ax.plot(x, [fchapeau(K, h, xi) for xi in x], color=[
                col1, col2, col3, col4][i], label=noms[i])
    ax.plot(x, reference(x), color=colref, label=noms[4]) # la densité de référence
    ax.set_xlabel('x')
    ax.set_ylabel('Densité')
    ax.set_title('Estimation de la densité avec différents noyaux (h='+str(h)+' n='+str(n)+')')
    ax.legend()
    # save figure to file
    plt.savefig('P1-BRETAGNOLLESPELOUTIERQ'+str(j)+g+'.png', dpi=300)

 
Allplotfchapeauh2(xmin, xmax, pas, col1, col2, col3, col4, colref,h)


#Question 6
j=6 

def Allplotfchapeauh1(xmin,xmax,pas,col1,col2,col3,col4,colref,h):
    Allplotfchapeauh2(xmin, xmax, pas, col1, col2, col3, col4, colref,h=1)
    
Allplotfchapeauh1(xmin,xmax,pas,col1,col2,col3,col4,colref,h)

#Qualitativement, est-ce que l’estimation diffère plus
#lorsque l’on fait varier le noyau utilisé ou la fenêtre h utilisée ?
#Réponse: l'estimation diffère plus lorsque l'on fait varier la fenêtre h utilisée.
# Plus h est grand, plus les densités sont applatis.



#Question 7
j=7
g='_1'
n=10
Allplotfchapeauh2(xmin, xmax, pas, col1, col2, col3, col4, colref, h=2)

g='_2'
n=10
Allplotfchapeauh2(xmin, xmax, pas, col1, col2, col3, col4, colref, h=1)

g='_3'
n=1000
Allplotfchapeauh2(xmin, xmax, pas, col1, col2, col3, col4, colref, h=2)

g='_4'
n=1000
Allplotfchapeauh1(xmin, xmax, pas, col1, col2, col3, col4, colref, h=1)

#On remarque que lorsque n est grand et h est petit, l'estimation est meilleure.

#Question 8    
j=8

def SCE(funct, h, f):
	t = np.arange(-5, 5, 10/500)
	return np.sum(([fchapeau(funct, h, ti) for ti in t]-f(t))**2)

erreurQuadratiqueK1 = SCE(K1, 0.5, reference)
erreurQuadratiqueK2 = SCE(K2, 0.5, reference)
erreurQuadratiqueK3 = SCE(K3, 0.5, reference)
erreurQuadratiqueK4 = SCE(K4, 0.5, reference)

print("Erreur quadratique de K1="+str(erreurQuadratiqueK1))
print("Erreur quadratique de K2="+str(erreurQuadratiqueK2))
print("Erreur quadratique de K3="+str(erreurQuadratiqueK3))
print("Erreur quadratique de K4="+str(erreurQuadratiqueK4))



#Question 9
j=9

def lemeilleurh(funct, f):
	h = np.arange(0.01, 2, 0.01)
	return np.argmin([SCE(funct, hi, f) for hi in h])/100

meilleurhK1=lemeilleurh(K1, reference)
meilleurhK2=lemeilleurh(K2, reference)
meilleurhK3=lemeilleurh(K3, reference)
meilleurhK4=lemeilleurh(K4, reference)

print("Meilleur h pour K1="+str(meilleurhK1))
print("Meilleur h pour K2="+str(meilleurhK2))
print("Meilleur h pour K3="+str(meilleurhK3))
print("Meilleur h pour K4="+str(meilleurhK4))
 
#Question 10 
j=10
#Définir alors une fonction qui représente graphiquement les quatres estimations de densité pour
#ces quatre noyaux avec les fenêtre obtenue via la fonction lemeilleurh. Définir pour ce faire, la
#fonction Allplotfchapeauhoptimal

#

'''
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
'''
