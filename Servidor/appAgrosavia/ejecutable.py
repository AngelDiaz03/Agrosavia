# <!-- 
# # ____________________________________________________________________________ 
#     ____   ____              ______             _                   ___ ____     
#    / __ \ / __ \            /_  __/___   _____ (_)_____            <  // __ \    
#   / / / // /_/ /  ______     / /  / _ \ / ___// // ___/  ______    / // /_/ /    
#  / /_/ // _, _/  /_____/    / /  /  __/(__  )/ /(__  )  /_____/   / / \__, /     
# /_____//_/ |_|             /_/   \___//____//_//____/            /_/ /____/      
                                                                                 
# # ____________________________________________________________________________ 
# # index.html
# # Author: Daniela Rojas Lozano, ddp.rojas11@uniandes.edu.co
# # Configuración, atributos, diseño de interfaz.
# # ____________________________________________________________________________                                                                       
# #
# -->

import numpy as np
from flask import Flask, request, render_template, json, Response
import time
import socket
import threading
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


app = Flask(__name__)


global numeroAnimalMobile
numeroAnimalMobile = 1
global NUMERO_MAXIMO_MODULOS
NUMERO_MAXIMO_MODULOS = 3

global ESTADO_MODULOS
ESTADO_MODULOS = []

global ACTIVIDAD_ACTUAL
global datosCompletos 
global datosTemporales 
ACTIVIDAD_ACTUAL = [] ## una actividad para cada animal con dispositivo
datosCompletos =[]
datosTemporales =[]
global tiemposEstadisticosTotales 
global tiempo_inicio_por_modulo 
global tiempo_ultima_actualizacion_por_modulo
global tamañoDatosTemporales
tiemposEstadisticosTotales = []
tiempo_ultima_actualizacion_por_modulo = []
tiempo_inicio_por_modulo = []
for modulo in  range(0,NUMERO_MAXIMO_MODULOS):
	ACTIVIDAD_ACTUAL.append(2)
	ESTADO_MODULOS.append(0)
	datosCompletos.append(  ['','','','','','','','','',''] )
	tiemposEstadisticosTotales.append( [ 0.000, 0.000 , 0.000 , 0.000 ])
	tiempo_ultima_actualizacion_por_modulo.append(time.time())
	tiempo_inicio_por_modulo.append('0')

datosTemporales = np.zeros((len(datosCompletos) , 1,6)).tolist()
tamañoDatosTemporales = np.zeros((NUMERO_MAXIMO_MODULOS, 1)).tolist()

global LISTA_ACTIVIDADES
global LISTA_LINKS
#### Comiendo = 0
#### Rumiando = 1
#### Nada = 2
LISTA_ACTIVIDADES = ['Comiendo', 'Rumiando', 'Nada']
LISTA_LINKS = ['comiendo.png', 'rumia.png', 'caminando.png', 'nada.png']


lock = threading.Lock()

global TIEMPO_INICIO
TIEMPO_INICIO = time.time() #Variable que guarda el tiempo en el que el servidor empezó a correr
global fechaYhora
FECHA_Y_HORA = "" #Variable para etiquetar los datos con fechas y horas
global nuevaLineaDatos
nuevaLineaDatos = [0,0,0,0,0,0,0,0,0,0,""]

###SWITCH MLC
global estadoLEDMLC
estadoLEDMLC = 0


### hilo para recivir los datos por sockets
def ThreadActualizarSocket():
	global TIEMPO_INICIO
	global ACTIVIDAD_ACTUAL
	global datosCompletos 
	global datosTemporales 
	global nuevaLineaDatos
	global ESTADO_MODULOS
	global NUMERO_MAXIMO_MODULOS
	global tiempo_inicio_por_modulo
	print("ThreadActualizarSocket Started... ")
	#UDP_IP = '192.168.0.13'
	UDP_IP = '192.168.0.8'
	UDP_PORT = 9001 ## Este puerto debe coincidir con el configurado en el módulo wifi para el envío de datos
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.bind((UDP_IP, UDP_PORT))
	print("____________________________________________")
	print("Puerto Abierto")
	print("____________________________________________")	


	print('[TCP][SOCKET] ')
	print('[TCP][SOCKET] INIT LOOP')
	print('[TCP][SOCKET] ')
	print('[TCP][SOCKET] HOST')
	print(UDP_IP)
	print('[TCP][SOCKET] PORT')
	print(UDP_PORT)
	print('[TCP][SOCKET] ')

	while True:
		lock.acquire()
		data, addr = s.recvfrom(1024)
		raw = data
		data = data.decode()

		print('[TCP][SOCKET] PACKET')
		print(data)
		print(raw)

		file = open("./resultados/datosAccel/completos"+str(TIEMPO_INICIO)+".txt","a") #Se activa (crea) el archivo para guardar (escribir) un nuevo dato
		try:
			ArrayData = data.split("#")
			Ax = float(ArrayData[0])
			Ay = float(ArrayData[1])
			Az = float(ArrayData[2])
			Gx = float(ArrayData[3])
			Gy = float(ArrayData[4])
			Gz = float(ArrayData[5])
			IdClient = int(ArrayData[6])-1
			IdClient = 2
			NumeroPaquete = float(ArrayData[7])
			fechaYhora = str(time.strftime("%c"))

			Ch4 = float(ArrayData[8])
			
			print(' ')
			print('Ch4')
			print(Ch4)
			print('NumeroPaquete')
			print(NumeroPaquete)
			print(' ')

			print(' ')
			nuevaLineaDatos = [ Ax,Ay,Az,Gx,Gy,Gz,Ch4, IdClient,NumeroPaquete, ACTIVIDAD_ACTUAL[IdClient], fechaYhora]
			print ("linea", nuevaLineaDatos)
			print(' ')
			
			datosCompletos[IdClient].append(nuevaLineaDatos)
			datosTemporales[IdClient].append(nuevaLineaDatos[0:7])


			ESTADO_MODULOS[IdClient] = 1
			if ( tiempo_inicio_por_modulo[IdClient]== '0'):
				tiempo_inicio_por_modulo[IdClient] = time.ctime(time.time())
				tiempo_ultima_actualizacion_por_modulo = time.time()
			file.write( ( fechaYhora + "	 %.1f	 %.5f	 %.5f	 %.5f	 %.5f	 %.5f	 %.5f	 %.5f	 %.5f  "%(NumeroPaquete, IdClient, Ax, Ay, Az, Gx, Gy, Gz, Ch4) ) + "	" + str(ACTIVIDAD_ACTUAL[IdClient]) + "	\n" )
			file.close() #Cada vez que el servidor recibe un dato lo guarda adecuamente en el archivo plano de texto
					  	 #para evitar perdidas de datos
		except Exception as e:
			print(e)
			print("Error en dato")
		lock.release()
threading.Thread(target=ThreadActualizarSocket).start()

### hilo para realizar predición de estados
def ThreadMLC():
	global estadoLEDMLC
	global datosTemporales 
	global ACTIVIDAD_ACTUAL
	global NUMERO_MAXIMO_MODULOS
	global ESTADO_MODULOS
	global tamañoDatosTemporales
	global tiemposEstadisticosTotales 
	global tiempo_ultima_actualizacion_por_modulo
	print("ThreadMLC Started ...")
	try:
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("model.h5")
		print("Loaded model from disk")
		loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	except Exception as inst:
	 	print("No se cargó el modelo")
	 	print(type(inst))    # la instancia de excepción
	 	print(inst.args)     # argumentos guardados en .args
	 	print(inst)          # __str__ permite imprimir args directamente,
	else:
	 	print("____________________________________________")
	 	print("MODELO CARGADO")
	 	print("____________________________________________")	


	while True:
		longitudTemporalModuloMayor=0
		for i in range(0,NUMERO_MAXIMO_MODULOS):
			x = len(datosTemporales[i])
			#### actualiza los estados de los modulos de acuerdo al tamaño de los temporales!
			if (tamañoDatosTemporales[i]==x):
				ESTADO_MODULOS[i]=0
			else:
				tamañoDatosTemporales[i]=x
				ESTADO_MODULOS[i]=1

			### guarda la longitud mayor para despues saber si hay datos suficientes para hacer la prediccion
			if (x>longitudTemporalModuloMayor):
				longitudTemporalModuloMayor=x

		if(estadoLEDMLC==1):
			if(longitudTemporalModuloMayor>= 200):
				for animali in range(0,NUMERO_MAXIMO_MODULOS):
					x = len(datosTemporales[animali])
					if (x>= 100): ### si no esta vacio 
						### modulo activo:
						ESTADO_MODULOS[animali] = 1
						print("Realizando predicción . . .")
						lock.acquire()
						### se realiza la predicion
						datosPredecir = datosTemporales[animali]
						datosPredecir = np.array( [datosPredecir[len(datosPredecir)-100:len(datosPredecir)] ]) ### Matriz 100 filas, 6 columnas
						#print(datosPredecir, datosPredecir.shape)
						datosPredecir = datosPredecir.reshape(1,100,6,1)
						res = loaded_model.predict(datosPredecir)
						#print('res', res)
						predecida = int(np.argmax(res[0]))
						#print('predecida', predecida)
						ACTIVIDAD_ACTUAL[animali]= predecida
						#print('ACTIVIDAD_ACTUAL',ACTIVIDAD_ACTUAL)
						lock.release()
					else:
						ESTADO_MODULOS[animali] = 0
				datosTemporales = np.zeros((len(datosCompletos) , 1,6)).tolist()

		#print("esperar")
		time.sleep(10)

threading.Thread(target=ThreadMLC).start()

@app.route('/')
def indexTemplate():
	global ACTIVIDAD_ACTUAL
	global NUMERO_MAXIMO_MODULOS
	global LISTA_ACTIVIDADES
	global LISTA_LINKS
	global numeroAnimalMobile
	LISTA = zip(LISTA_ACTIVIDADES, LISTA_LINKS, range(0,len(LISTA_ACTIVIDADES)))
	ListaModulos = range(0, len(ACTIVIDAD_ACTUAL))
	return render_template( 'index.html' , ListaModulos=ListaModulos,NUMERO_MAXIMO_MODULOS=NUMERO_MAXIMO_MODULOS,LISTA=LISTA  )

@app.route('/actualizar_estado', methods = ['POST'])
def actualizar_estado():
	global ACTIVIDAD_ACTUAL
	global NUMERO_MAXIMO_MODULOS
	for modulo in  range(0,NUMERO_MAXIMO_MODULOS): 
		ACTIVIDAD_ACTUAL[modulo] = int(request.form['estadoVaca'+str(modulo)])
	#print('ACTIVIDAD_ACTUAL',ACTIVIDAD_ACTUAL)
	##print("xxxxxxxxxxxxxxxxxxx")
	##print('ACTIVIDAD_ACTUAL',ACTIVIDAD_ACTUAL )
	return ""

### Recibe la información del switch MLC, para así comenzar a predecir
@app.route('/switchMLC', methods = ['POST'])
def switchMLC():
	global estadoLEDMLC
	estadoLEDnuevo = request.form['led']
	#print("la nueva accion del LED es : "  + estadoLEDnuevo)
	if(estadoLEDnuevo=='true'):
		estadoLEDMLC=1
	elif(estadoLEDnuevo=='false'):
		estadoLEDMLC=0
	return ""


### Envia la información de los estados de conexion de los modulos
@app.route('/actualizarEstadoModulos', methods=['POST'])
def actualizarEstadoModulos():
	#enviar información al cliente
	global NUMERO_MAXIMO_MODULOS
	global ESTADO_MODULOS
	global datosTemporales
	ListaModulos = range(0, len(ACTIVIDAD_ACTUAL))
	for modulo in ListaModulos:
		if(len(datosTemporales[modulo])==0):
			ESTADO_MODULOS[modulo] = 0

	return json.dumps({'NUMERO_MAXIMO_MODULOS':NUMERO_MAXIMO_MODULOS,'ESTADO_MODULOS':ESTADO_MODULOS, 'ListaModulos':list(ListaModulos)});

### Envia la información de las graficas.
@app.route('/actualizarGraficas', methods=['POST'])
def actualizarGraficas():
	#enviar información al cliente
	global NUMERO_MAXIMO_MODULOS
	global TIEMPO_INICIO
	global nuevaLineaDatos ##[ Ax,Ay,Az,Gx,Gy,Gz,IdClient,NumeroPaquete,ACTIVIDAD_ACTUAL[IdClient], fechaYhora]
	modulo = int(nuevaLineaDatos[7])
	Ax = nuevaLineaDatos[0]
	Ay = nuevaLineaDatos[1]
	Az = nuevaLineaDatos[2]
	Gx = nuevaLineaDatos[3]
	Gy = nuevaLineaDatos[4]
	Gz = nuevaLineaDatos[5]
	Ch4 = nuevaLineaDatos[6]
	tiempo = time.time()-TIEMPO_INICIO
	ListaModulos = range(0, len(ACTIVIDAD_ACTUAL))
	return json.dumps({'Ax':Ax ,'Ay':Ay ,'Az':Az ,'Gx':Gx ,'Gy':Gy ,'Gz':Gz , 'Ch4':Ch4, 'tiempo':tiempo, 'modulo':modulo,'NUMERO_MAXIMO_MODULOS':NUMERO_MAXIMO_MODULOS, 'ListaModulos':list(ListaModulos), 'nuevaLineaDatos':nuevaLineaDatos});

def calcularTiempos():
	global NUMERO_MAXIMO_MODULOS
	global ACTIVIDAD_ACTUAL
	global tiemposEstadisticosTotales 
	global tiempo_inicio_por_modulo 
	global tiempo_ultima_actualizacion_por_modulo
	global ESTADO_MODULOS
	nptiempoEstadisticos = np.array(tiemposEstadisticosTotales)
	for modulo_i in range(0,NUMERO_MAXIMO_MODULOS):
		tiempoActual = time.time()
		if not (ESTADO_MODULOS[modulo_i] == 0):
			tiempoTranscurridoHoras = (((tiempoActual - tiempo_ultima_actualizacion_por_modulo[modulo_i])/3600))
			nptiempoEstadisticos[int(modulo_i),int(ACTIVIDAD_ACTUAL[modulo_i])] = (tiempoTranscurridoHoras +  nptiempoEstadisticos[int(modulo_i),int(ACTIVIDAD_ACTUAL[modulo_i])])
		tiempo_ultima_actualizacion_por_modulo[modulo_i] = tiempoActual
	tiemposEstadisticosTotales = nptiempoEstadisticos.tolist()
### Envia la información de los estados de los animales
@app.route('/actualizarEstadosVacas', methods=['POST'])
def actualizarEstadosVacas():
	#enviar información al cliente
	global NUMERO_MAXIMO_MODULOS
	global ACTIVIDAD_ACTUAL
	global tiemposEstadisticosTotales 
	global tiempo_inicio_por_modulo 
	calcularTiempos()
	nptiempoEstadisticos = np.array(tiemposEstadisticosTotales)
	tiemposComer = [ '%.2f' % elem for elem in nptiempoEstadisticos[:,0].tolist() ]
	tiemposRumia = [ '%.2f' % elem for elem in nptiempoEstadisticos[:,1].tolist() ]
	tiemposNada = [ '%.2f' % elem for elem in nptiempoEstadisticos[:,2].tolist() ]
	#tiemposNada = [ '%.2f' % elem for elem in nptiempoEstadisticos[:,3].tolist() ]
	ListaModulos = range(0, len(ACTIVIDAD_ACTUAL))
	np.savetxt("./resultados/tiemposEstadisticosTotales", tiemposEstadisticosTotales, newline=" \n ")
	#print("tiempos",tiemposComer,tiemposRumia,tiemposCaminar,tiemposNada)
	return json.dumps({'tiempo_inicio_por_modulo':tiempo_inicio_por_modulo,'tiemposNada':tiemposNada,'tiemposRumia':tiemposRumia,'tiemposComer':tiemposComer,'ACTIVIDAD_ACTUAL':ACTIVIDAD_ACTUAL,'NUMERO_MAXIMO_MODULOS':NUMERO_MAXIMO_MODULOS, 'ListaModulos':list(ListaModulos)});


if __name__ == '__main__':
	app.run(host='0.0.0.0',debug = False ,port= 9001, use_reloader=False, threaded=True)





# <!-- # ____________________________________________________________________________                                                                       
# # ______            _      _        ______      _           
# # |  _  \          (_)    | |       | ___ \    (_)          
# # | | | |__ _ _ __  _  ___| | __ _  | |_/ /___  _  __ _ ___ 
# # | | | / _` | '_ \| |/ _ \ |/ _` | |    // _ \| |/ _` / __|
# # | |/ / (_| | | | | |  __/ | (_| | | |\ \ (_) | | (_| \__ \
# # |___/ \__,_|_| |_|_|\___|_|\__,_| \_| \_\___/| |\__,_|___/
# #                                             _/ |          
# #                                            |__/           
# # ____________________________________________________________________________                                                                       
# # -->

