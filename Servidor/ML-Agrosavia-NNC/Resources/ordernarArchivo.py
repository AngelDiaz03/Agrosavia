# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                                
#     __  _____       ___    ____  ____              ___   ____ ___   ____ 
#    /  |/  / /      /   |  / __ \/ __ \            |__ \ / __ \__ \ / __ \
#   / /|_/ / /      / /| | / /_/ / /_/ /  ______    __/ // / / /_/ // / / /
#  / /  / / /___   / ___ |/ ____/ ____/  /_____/   / __// /_/ / __// /_/ / 
# /_/  /_/_____/  /_/  |_/_/   /_/                /____/\____/____/\____/  
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                                
# modelTraining.py 
# Author: Daniela Rojas 
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                               
   

   

import numpy as np
separador=","
f = open("completos.csv", "r")
datos = []
for x in f:
	datos.append(x.split(separador))

datosnp = np.array(datos)
###idMaxModulos = int(max(datosnp[:,2]))
datosModulo0 = []
datosModulo1 = []
datosModulo2 = []
cantidadTotalDatos = datosnp.shape[0]
for i in range(0, cantidadTotalDatos):
	if(int(datosnp[i,2])==0):
		datosModulo0.append(datosnp[i,:].tolist())
	elif (int(datosnp[i,2])==1):
		datosModulo1.append(datosnp[i,:].tolist())
	elif (int(datosnp[i,2])==2):
		datosModulo2.append(datosnp[i,:].tolist())

datosModulo0 = np.array(datosModulo0)
l0 = datosModulo0.shape[0]
datosModulo1 = np.array(datosModulo1)
l1 = datosModulo1.shape[0]
datosModulo2 = np.array(datosModulo2)
l2 = datosModulo2.shape[0]

longitudMayor = 0
if(l1>l0 and l1>l2):
	longitudMayor = l1
elif (l2>l0 and l2>l1):
	longitudMayor = l2
else:
	longitudMayor = l0

datosOrdenados = []
for i in range(0,longitudMayor):
	fila = []
	if (i<l0 ):
		fila = datosModulo0[i,:].tolist()
	else:
		fila = ['','','','','','','','','','']
	if (i<l1):
		fila = np.concatenate((fila, datosModulo1[i,:].tolist() ) )
	else:
		fila = np.concatenate((fila, ['','','','','','','','','',''] ) )
	if (i<l2):
		fila = np.concatenate((fila, datosModulo2[i,:].tolist() ) )
	else:
		fila = np.concatenate((fila, ['','','','','','','','','',''] ) )

	datosOrdenados.append(fila)
datosOrdenados = np.array(datosOrdenados)

print("cantidadTotalDatos",cantidadTotalDatos)
print("datosModulo0",l0)
print("datosModulo1",l1)
print("datosModulo2",l2)
print('longitudMayor', longitudMayor)
print('datosOrdenados', datosOrdenados.shape)


import csv

myFile = open('ordenadosModulos.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(datosOrdenados)
     
print("Writing complete")


