# Tarea3

#### Parte 1.

~~~python
print("Parte1")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#Leer los datos
xy = np.genfromtxt("xy.csv", dtype=float, delimiter=",", skip_header=1, usecols=range(1,22))
print("Vector yPMF")
yPMF=np.sum(xy, axis=0)
print(yPMF)
print("Vector xPMF")
xPMF=np.sum(xy, axis=1)
print(xPMF)
#Encontrar la mejor curva de ajuste
x=np.linspace(5, 15, 11)

plt.plot(x, xPMF)
print("Se observa que la mejor curva de ajuste, hace referencia a una Distribución Gaussiana")

#Se encuentran los parámetros


def gaussiana(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))
x=np.linspace(5,15,11)

param, _ = curve_fit(gaussiana,  x,xPMF)
print("Parámetros de la curva de mejor ajuste = ",param)
~~~

#### Parte 2.
2. Asumir independencia de X y Y Analíticamente, ¿cuál es entonces la expresión de la función de densidad conjunta que modela los datos?
Con los parámetros encontrados en el código de las funciones de PMF de X y Y, se puede obtener las funciones para luego ser multiplicadas y obtener la función de densidad conjunta.

Parámetros de ajusteX = [9.90484381 3.29944287] 
Parámetros de ajusteY= [15.0794609 6.02693776]

  2.1 De manera analítica y con los parámetros de mu(media) y sigma(desviación) se obtiene para la PMF de X, la siguiente función:
f(xPMF)= 1/(np.sqrt(2np.pi3.2992))*np.exp(-(x-9.904)2/(2*3.299**2))

  2.2 De manera analítica y con los parámetros de mu(media) y sigma(desviación) se obtiene para la PMF de X, la siguiente función:
f(yPMF)= 1/(np.sqrt(2np.pi6.0262))*np.exp(-(x-15.07)2/(2*6.026**2))

2.3 Por lo tanto la función de densidad conjunta que corresponde a fx,y(X,Y)=fx(X)*fy(Y), es por esto que:
La función de densidad conjunta es: 0.008*np.exp(-((((x-9.904)2)/15.7)+(((y-15.04)2)/59.06)))

Es importante mencionar que la dependencia estadística de variables aleatorias múltiples plantea que la la probabilidad en que suceda en forma conjunta un evento con otro es equivalete a que suceda lo siguiente
P{X < ó = x, Y< ó = y }=P{X < ó = x}P{Y < ó = y } O bien: fx,y(X,Y)=fx(X)fy(Y)


#### Parte 3

~~~python
print("Parte3")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
xyp=pd.read_csv("xyp.csv")
xyp["resultado1"]=xyp["x"]*xyp["y"]*xyp["p"]
#hacer la suma
correlación=np.sum(xyp["resultado1"], axis=0)
print("La correlación es=", correlación)
mediax=np.mean(xyp['x'])
mediay=np.mean(xyp['y'])
print("Media de x =",mediax)
print("Media de y =",mediay)
#se realiza la fórmula
xyp['resultado2'] = (xyp['x']-mediax)*(xyp['y']-mediay)*xyp['p']
#Hacer la suma 
covarianza=np.sum(xyp['resultado2'] , axis=0)
print("La covarianza es = ",covarianza)
varianzay=np.std(xyp["y"])
varianzax=np.std(xyp["x"])
#Coeficiente de correlación
#Calcular la varianza de x y y
#Calcular la "multiplicacion"
xyp['resultado3'] = (xyp['resultado2'])/(varianzax*varianzay)
#Hacer la suma
coefcorre=np.sum(xyp['resultado3'] , axis=0)
#imprimir el resultado
print('El coeficiente de correlación es =', coefcorre)
~~~

#### Parte 4
~~~python
print("Parte4")
#Graficar las funciones de densidad marginales (2D), la función de densidad conjunta (3D)

#Gráfica función de densidad (2D)
#Se debe encontrar las gráficas de la PMF de "x" y de "y"
print("Funciones de densidad marginales  2D")


yF=np.sum(xy, axis=0)
plt.plot(yF)   #Representada en color azul  
plt.show
xF=np.sum(xy, axis=1)
plt.plot(xF)    #Representada en color naranja
plt.title("PMF de X y Y")
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.show
~~~

### Parte 5
~~~python
print("Gráfica función de densidad conjunta (3D)")
#Para en contrar la función de densidad conjunta, se sabe que es la multiplicación de ambas funciones.
#Por esto se debe encontrar los parámetros de ambas xPMF y de yPMF, multiplicarlas y encontrar la fucnión de densidad conjunta
#Parámetros de xPMF
from scipy.optimize import curve_fit
def gaussiana(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))
x=np.linspace(5,15,11)
param, _ = curve_fit(gaussiana,  x,xPMF)
print("Parámetros de ajusteX = ",param)
#Parámetros de yPMF
y=np.linspace(5,25,21)
param, _= curve_fit(gaussiana, y, yPMF)
print("Parámetros de ajusteY=", param)
#Con estos parámetros de mu y sigma, se encuentran ambas funciones gaussianas y se multiplican 
#Encontrando así una fucnión z de manera analítica, que se necesita para la función de densidad conjunta, a como se muestra en el siguiente código
from matplotlib.pyplot import *
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
x = np.linspace(0, 15, 100)
y = np.linspace(0, 25, 100)
x,y = np.meshgrid(x,y)
z = 0.008*np.exp(-((((x-9.904)**2)/15.7)+(((y-15.04)**2)/59.06)))

fig = plt.figure(figsize=(7,7))

ax=fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(x,y,z, rstride=2, cstride=2, cmap='Blues')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.set_title('Funcion de Distribucion Conjunta')
plt.show()
~~~
