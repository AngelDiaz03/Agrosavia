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
   

import time
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import math
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import classification_report, confusion_matrix

### 
COLUMNA_FECHA = 0
COLUMNA_X1 = 3
COLUMNA_X2 = 4
COLUMNA_X3 = 5
COLUMNA_X4 = 6
COLUMNA_X5 = 7
COLUMNA_X6 = 8
COLUMNA_LABEL = 9
feat_labels = [ "X1" , "X2", "X3" , "X4", "X5", "X6"]


#optionsDataProcessing = ["Use all data available", "Use 1 data/sec", "Use a dynamic window of data"] #0,1,2
#optionsAlgorithm = ["Random Forest", "Neural Network", "Convolutional Neural Network"] #0,1,2

# Random Forest:
#configuration {'% Train': '70', '# of Trainings': '3', 'Hyperparameter optimization': True, 'Active Module IDs': [1, 2]}
# neural network
# configuration {'% Train': '0', 'Total layers': 1, 'Layers': {'Layer 1': ['0', 'Sigmoidal']}, 'Active Module IDs': [1, 2]}
def trainModel(pDataFrames, pDataProcessing, pAlgorithm, configuration):
	time.sleep(2)
	print("--------------------------------------------------------")
	print("Start: Training Model")
	print("--------------------------------------------------------")
	#print("pDataFrames",pDataFrames)
	resultsToReport = { }

	activeModules = configuration['Active Module IDs']
	#print("activeModules",activeModules)

	X = []
	Y = []
	##########################################
	### functions  optionsDataProcessing
	##########################################
	if pDataProcessing == 0:
		#### "Use all data available"
		X, Y = useAllDataAvailable(pDataFrames, activeModules)
	elif pDataProcessing ==1:
		#### "Use 1 data/sec"
		X, Y = averagePerSecond(pDataFrames, activeModules)
	elif pDataProcessing ==2:
		#### "Use a dynamic window of data"
		X, Y = dynamicWindow4(pDataFrames, activeModules)
	elif pDataProcessing ==3:
		#### "Use a dynamic window of data"
		X, Y = dynamicWindow2(pDataFrames, activeModules)
	
	#print("X", "Y", X, Y)

	### it is mandatory convert the labels to numbers.
	Y = convertLabelsToNumbers(Y)
	longitudX = len(X)
	longitudY = len(Y)
	if not longitudY==longitudX:
		if longitudY<longitudX:
			X = X[0:longitudY,:] 
			print("xxxxxxxxxxxxxxcxxxxxxxxxxxx")
		elif longitudX<longitudY:
			Y = Y[0:longitudX,:] 
			print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmm")

	longitudX = len(X)
	longitudY = len(Y)
	if longitudY==longitudX:
		print("Hecho")
		
	print("Data is ready")
	print("--------------------------------------------------------")
	resultsToReport['X'] = X
	resultsToReport['Y'] = Y


	##########################################
	### functions pAlgorithm
	##########################################
	# if pAlgorithm == 0:
	# 	#### Random Forest"
	# 	resultsToReport = algorithmRandomForest(configuration, X, Y, resultsToReport)
	# elif pAlgorithm ==1:
	# 	#### "Neural Network"
	# 	resultsToReport = algorithNeuralNetwork(configuration, X, Y,resultsToReport)
	# elif pAlgorithm ==2:
	# 	#### "Convolution Neural Network"
	resultsToReport = algorithConvolutionNeuralNetwork(configuration, X, Y,resultsToReport)
	#elif pAlgorithm ==2:
		#### "Convolutional Neural Network"
		#resultsToReport = algorithConvolutionNeuralNetwork(configuration, X, Y,resultsToReport)
		
	
	
	print("--------------------------------------------------------")
	print("Model Training finished successful ")
	print("--------------------------------------------------------")
	print("resultsToReport", resultsToReport)
	print("resultsToReport len", len(resultsToReport))

	return resultsToReport

####################################################################################
### label string to number
####################################################################################
# 0 = Nada
# 1 = Caminando  
# 2 = Rumiando(Trotadora)
# 3 = Comiendo(Elíptica)
# 4 = Agua(Escaladora)
def convertLabelsToNumbers(arrayLabelsString):
	arrayLabelsInt = []

	for dato in arrayLabelsString:
		arrayLabelsInt.append(int(dato))

	arrayLabelsInt = np.array(arrayLabelsInt)
	return arrayLabelsInt


####################################################################################
### functions  optionsDataProcessing
####################################################################################
def useAllDataAvailable(pDataFrames, activeModules):
	print("pDataProcessing  0 : Use all data available")
	nModule = len(pDataFrames)
	nActivateModules = len(activeModules)

	#dataToUse = ['x1','x2','x3','x4','x5','x6'] don't care modules
	dataToUse = np.array([[0, 0, 0 , 0 , 0 , 0 ]])
	labelToUse = [""]
	
	for i in range(0,nActivateModules):
		### id-1, list in python begin in 0
		idModulei = activeModules[i]-1
		datosActiveModulei = np.array(pDataFrames[idModulei])
		### add data for activateModule i 
		dataToUse = np.concatenate ( (dataToUse,   datosActiveModulei[ : , COLUMNA_X1:COLUMNA_LABEL ] ) )
		labelToUse =  np.concatenate(  ( labelToUse,  datosActiveModulei[ : , COLUMNA_LABEL ] ) )

	## delete the first row of zeros
	dataToUse = dataToUse[1:dataToUse.shape[0], :]
	labelToUse =  labelToUse[1:labelToUse.shape[0]]
	print("dataToUse shape", dataToUse.shape)
	print("labelToUse", labelToUse.shape)
	print("--------------------------------------------------------")
	print("Done: --> Used all data available of active modules")
	print("--------------------------------------------------------")
	return(dataToUse,labelToUse)

def averagePerSecond(pDataFrames, activeModules):
	print("pDataProcessing 1: Use 1 data/sec")
	nModule = len(pDataFrames)
	nActivateModules = len(activeModules)

	#dataToUse = ['x1','x2','x3','x4','x5','x6'] don't care modules
	dataToUse = np.array([[0, 0, 0 , 0 , 0 , 0 ]])
	labelToUse = [""]
	
	for i in range(0,nActivateModules):
		### id-1, list in python begin in 0
		idModulei = activeModules[i]-1
		datosActiveModulei = np.array(pDataFrames[idModulei])
		
		### one by one second have a average... 
		segundoAnterior = ""
		promedio = datosActiveModulei[ 0 , COLUMNA_X1:COLUMNA_LABEL ]
		data_Modulei_Average = []
		label_Modulei_STRING = []
		for informacion in datosActiveModulei :
			segundoActual = informacion[ COLUMNA_FECHA ]

			if segundoActual == segundoAnterior:
				promedio = np.average( [informacion[  COLUMNA_X1:COLUMNA_LABEL ], promedio]   , axis=0) 
			else:
				data_Modulei_Average.append(  np.around ( promedio.tolist(), decimals = 3  ).tolist()   )
				label_Modulei_STRING.append(informacion[COLUMNA_LABEL] )
				segundoAnterior = segundoActual
				promedio = informacion[ COLUMNA_X1:COLUMNA_LABEL ]

		print("Active module number ",i," had a record for " , len(label_Modulei_STRING), "seconds.")

		### add data for activateModule i
		dataToUse = np.concatenate ( (dataToUse,  data_Modulei_Average  ) )
		labelToUse =  np.concatenate(  ( labelToUse, label_Modulei_STRING ) )


	## delete the first row of zeros
	dataToUse = dataToUse[1:dataToUse.shape[0], :]
	labelToUse =  labelToUse[1:labelToUse.shape[0]]

	# print("dataToUse", dataToUse)
	print("dataToUse shape", dataToUse.shape)
	#print("labelToUse", labelToUse)
	print("labelToUse shape", labelToUse.shape)
	print("--------------------------------------------------------")
	print("Done: --> Used 1 data/sec")
	print("--------------------------------------------------------")
	return(dataToUse,labelToUse)

def dynamicWindow4(pDataFrames, activeModules):
	print("pDataProcessing 2: Use a dynamic window of data ")
	##### x. = x[t-1]+x[t]+x[t+1]+x[t+2]
	nModule = len(pDataFrames)
	nActivateModules = len(activeModules)
	#dataToUse = ['x1','x2','x3','x4','x5','x6'] don't care modules
	dataToUse = np.array([[0, 0, 0 , 0 , 0 , 0 ]])
	labelToUse = [""]
	
	for i in range(0,nActivateModules):
		### id-1, list in python begin in 0
		idModulei = activeModules[i]-1
		datosActiveModulei = np.array(pDataFrames[idModulei])
		lenDatosActiveModulei = len(datosActiveModulei)

		valoresPosicionAnterior = datosActiveModulei[ 0 , COLUMNA_X1:COLUMNA_LABEL ]

		data_Modulei = []
		label_Modulei_STRING = []
		for posicion in range(0,lenDatosActiveModulei-3):
			valoresPosicionActual = datosActiveModulei[ posicion , COLUMNA_X1:COLUMNA_LABEL ]
			valoresPosicionSiguiente = datosActiveModulei[ posicion+1 , COLUMNA_X1:COLUMNA_LABEL ]
			valoresPosicionSiguienteSiguiente = datosActiveModulei[ posicion+2 , COLUMNA_X1:COLUMNA_LABEL ]

			total = valoresPosicionActual + valoresPosicionSiguiente + valoresPosicionSiguienteSiguiente + valoresPosicionAnterior
			#print("total",total)
			data_Modulei.append(  np.around ( total.tolist(), decimals = 3  ).tolist()   )
			label_Modulei_STRING.append( datosActiveModulei[ posicion , COLUMNA_LABEL ])
			valoresPosicionAnterior = valoresPosicionActual

		print("Active module number ",i," had a record for " , len(label_Modulei_STRING), "times.")
		### add data for activateModule i
		dataToUse = np.concatenate ( (dataToUse,  data_Modulei  ) )
		labelToUse =  np.concatenate(  ( labelToUse, label_Modulei_STRING ) )


	## delete the first row of zeros
	dataToUse = dataToUse[2:dataToUse.shape[0], :]
	labelToUse =  labelToUse[1:labelToUse.shape[0]]

	# print("dataToUse", dataToUse)
	print("dataToUse shape", dataToUse.shape)
	# print("labelToUse", labelToUse)
	print("labelToUse shape", labelToUse.shape)
	print("--------------------------------------------------------")
	print("Done: --> Used window x[t-1]+x[t]+x[t+1]+x[t+2]")
	print("--------------------------------------------------------")
	return(dataToUse,labelToUse)

def dynamicWindow2(pDataFrames, activeModules):
	print("pDataProcessing 2: Use a dynamic window of data size 2 x[t]+x[t+1]")
	##### x. = x[t]+x[t+1]
	nModule = len(pDataFrames)
	nActivateModules = len(activeModules)
	#dataToUse = ['x1','x2','x3','x4','x5','x6'] don't care modules
	dataToUse = np.array([[0, 0, 0 , 0 , 0 , 0 ]])
	labelToUse = [""]
	
	for i in range(0,nActivateModules):
		### id-1, list in python begin in 0
		idModulei = activeModules[i]-1
		datosActiveModulei = np.array(pDataFrames[idModulei])
		lenDatosActiveModulei = len(datosActiveModulei)

		data_Modulei = []
		label_Modulei_STRING = []
		for posicion in range(0,lenDatosActiveModulei-2):
			valoresPosicionActual = datosActiveModulei[ posicion , COLUMNA_X1:COLUMNA_LABEL ]
			valoresPosicionSiguiente = datosActiveModulei[ posicion+1 , COLUMNA_X1:COLUMNA_LABEL ]

			total = valoresPosicionActual + valoresPosicionSiguiente 
			#print("total",total)
			data_Modulei.append(  np.around ( total.tolist(), decimals = 3  ).tolist()   )
			label_Modulei_STRING.append( datosActiveModulei[ posicion , COLUMNA_LABEL ])

		print("Active module number ",i," had a record for " , len(label_Modulei_STRING), "times.")
		### add data for activateModule i
		dataToUse = np.concatenate ( (dataToUse,  data_Modulei  ) )
		labelToUse =  np.concatenate(  ( labelToUse, label_Modulei_STRING ) )


	## delete the first row of zeros
	dataToUse = dataToUse[2:dataToUse.shape[0], :]
	labelToUse =  labelToUse[1:labelToUse.shape[0]]

	# print("dataToUse", dataToUse)
	print("dataToUse shape", dataToUse.shape)
	# print("labelToUse", labelToUse)
	print("labelToUse shape", labelToUse.shape)
	print("--------------------------------------------------------")
	print("Done: --> Used window x[t]+x[t+1]")
	print("--------------------------------------------------------")
	return(dataToUse,labelToUse)

####################################################################################
### functions pAlgorithm
####################################################################################
def algorithmRandomForest(configuration, X, Y, resultsToReport):
	# Random Forest:
	#configuration {'% Train': '70', '# of Trainings': '3', 'Hyperparameter optimization': True}
	#'Hyperparameter optimization': False, 'N-estimators': '0', 'Max features': '0', 'Min samples leaf': '0'}
	print("Start: Random Forest")
	percentageTrain = configuration['% Train']
	percentageTest = np.around(1-(int(percentageTrain)/100), decimals = 3)
	nTrainings = int(configuration['# of Trainings'])
	print("percentageTrain", percentageTrain, "  nTrainings", nTrainings, "percentageTest", percentageTest)
	optimization=configuration['Hyperparameter optimization']


	if optimization == True:
		## optimization search the best hiperparameter for the data.
		print("looking for parameters . . .")
			### repeat all, for each training
		for trainingN in range(1,nTrainings+1):
			print("--------------------------------------------------------")
			print("Trainning number ", trainingN)
			X, Y = shuffle(X,Y)
			X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=percentageTest, random_state=0)
			print('There are', X_train.shape[0], 'training data and',  X_test.shape[0], 'testing data.')
			print(np.vstack((np.unique(Y_train), np.bincount(Y_train))).T)
			### primero optimzar n_Estimators que debe ser un entero entre 0 y 160
			resultadosAllCasesOpti =[[0,0 ,0 ,0,0 ]]
			maximunAccuarance = 0
			for nArboles in range(5,70,5):
				for n_max_features in range(1,6):
					for n_min_samples_leaf in range(1,5):
						### training random forest . . . with the input parameters
						model_rf = RandomForestClassifier(n_estimators=nArboles, max_features=n_max_features, min_samples_leaf=n_min_samples_leaf,random_state=0, n_jobs=2)
						model_rf.fit(X_train, Y_train.ravel())
						y_pred = model_rf.predict(X_test)
						precision=accuracy_score(Y_test, y_pred)
						funcionObjetivo = 10000*precision-nArboles-n_max_features-n_min_samples_leaf
						resultadosAllCasesOpti = resultadosAllCasesOpti + [[funcionObjetivo, nArboles,n_max_features,n_min_samples_leaf,precision]]
					print(".")
				print("..")

			#print("resultadosAllCasesOpti ")
			#print(resultadosAllCasesOpti )
			resultadosAllCasesOpti=np.array(resultadosAllCasesOpti)
			
			funcionObjetivo= resultadosAllCasesOpti[:,0]
			filaMax = np.argmax(funcionObjetivo)
			MAXdata = resultadosAllCasesOpti[filaMax,:]

			resultsToReport['X_train_'+str(trainingN)]=X_train
			resultsToReport['X_test_'+str(trainingN)]=X_test
			resultsToReport['Y_train_'+str(trainingN)]=Y_train
			resultsToReport['Y_test_'+str(trainingN)]=Y_test
			resultsToReport['resultadosAllCasesOpti_'+str(trainingN)]=resultadosAllCasesOpti
			resultsToReport['Opti_nEstimators_'+str(trainingN)]=MAXdata[1]
			resultsToReport['Opti_max_feature_'+str(trainingN)]=MAXdata[2]
			resultsToReport['Opti_samples_leaf_'+str(trainingN)]=MAXdata[3]
			resultsToReport['OptimaxPrecision_'+str(trainingN)]=MAXdata[4]

			model_rf = RandomForestClassifier(n_estimators=MAXdata[1], max_features=MAXdata[2], min_samples_leaf=MAXdata[3],random_state=0, n_jobs=2)
			resultsToReport['OptiModel_'+str(trainingN)]= model_rf

	else:
			### repeat all, for each training
		for trainingN in range(1,nTrainings+1):
			print("--------------------------------------------------------")
			print("Trainning number ", trainingN)
	
			X, Y = shuffle(X,Y)
			X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=percentageTest, random_state=0)
			print('There are', X_train.shape[0], 'training data and',  X_test.shape[0], 'testing data.')
			print(np.vstack((np.unique(Y_train), np.bincount(Y_train))).T)

			### training random forest . . . with the input parameters
			print("training random forest . . .")
			nEstimators = int(configuration['N-estimators'])
			nMaxFeatures = int(configuration['Max features'])
			nMinSamplesLeaf = int(configuration['Min samples leaf'])
			print("nEstimators","nMaxFeatures","nMinSamplesLeaf",nEstimators,nMaxFeatures,nMinSamplesLeaf)
			
			model_rf = RandomForestClassifier(n_estimators=nEstimators, max_features=nMaxFeatures, min_samples_leaf=nMinSamplesLeaf,random_state=0, n_jobs=2)
			model_rf.fit(X_train, Y_train.ravel())
			#####################################################
			# Encontrar importancia de cada variable, y graficar
			print('Calculating importance of variables for prediction ...')
			importanciaVars=model_rf.feature_importances_
			print(np.around(importanciaVars, decimals=3))

			# Validation data prediction
			print("""
				Validation data prediction...""")
			y_pred = model_rf.predict(X_test)
			precision=accuracy_score(Y_test, y_pred)
			print("%.4f" %precision)

			# Matriz de confusion
			print("""
				Graph confusion matrix...
				""")
			tabla=pd.crosstab(Y_test.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
			print(tabla*100/len(y_pred))

			resultsToReport['X_train_'+str(trainingN)]=X_train
			resultsToReport['X_test_'+str(trainingN)]=X_test
			resultsToReport['Y_train_'+str(trainingN)]=Y_train
			resultsToReport['Y_test_'+str(trainingN)]=Y_test
			resultsToReport['y_pred_'+str(trainingN)]=y_pred
			resultsToReport['importanciaVars_'+str(trainingN)]=np.around(importanciaVars, decimals=3)
			resultsToReport['precision_'+str(trainingN)] = precision
			resultsToReport['Condusionmatrix_'+str(trainingN)] = tabla*100/len(y_pred)
			resultsToReport['modelRF'+str(trainingN)] = model_rf
	
	return resultsToReport

def algorithConvolutionNeuralNetwork(configuration, X, Y, resultsToReport):
	# neural network
	# configuration {'% Train': '0', 'Total layers': 1, 'Layers': {'Layer 1': ['0', 'Sigmoidal']}}
	print("pAlgorithm 2: Convolution Neural Network")
	percentageTrain = configuration['% Train']
	percentageTest = np.around(1-(int(percentageTrain)/100), decimals = 3)

	datosnp = np.array(X)
	#print("!1")
	cantidadTotalDatos = datosnp.shape[0]
	labelsnp = np.array(Y)
	#print("!2")
	#print("cantidadTotalDatos",cantidadTotalDatos)

	matrices=[]
	labels = []

	for i in range(0, cantidadTotalDatos-100,10):
		a = np.array(datosnp[i:i+100,:])
		a = a.astype(np.float)
		p = int(labelsnp[i] )
		matrices.append(a)
		labels.append(p)
		
	matrices = np.array(matrices)
	labels = np.array(labels)
	print('matrices', matrices.shape)
	print('labels', labels.shape)

	X_train, X_test, Y_train, Y_test = train_test_split(matrices, labels, test_size=percentageTest, random_state=0)
	xtrainShape = X_train.shape[0]
	ytrainShape = X_test.shape[0]
	print('There are', X_train.shape[0], 'training data and',  X_test.shape[0], 'testing data.')
	print(np.vstack((np.unique(Y_train), np.bincount(Y_train))).T)
	arregloCuentas = np.vstack((np.unique(Y_train), np.bincount(Y_train))).T
	print('X_train.shape',X_train.shape)
	X_train = X_train.reshape(X_train.shape[0],100,6,1)
	X_test = X_test.reshape(X_test.shape[0],100,6,1)
	print('X_train.shape',X_train.shape)


	#one-hot encode target column
	Y_train = to_categorical(Y_train)
	Y_test = to_categorical(Y_test)
	print(Y_train[0])

	totalLayers = configuration['Total layers']
	infoLayers = configuration['Layers']
	nNeuronsTodas = []
	nfunctionTodas = []
	for i in range(1,totalLayers+1):
		print("layer ", i)
		nNeuronsi = infoLayers[ str('Layer ' + str(i) ) ][0]
		print("neuronas", nNeuronsi )
		nNeuronsTodas.append(int(nNeuronsi))
		functionActivationi = infoLayers[ str('Layer ' + str(i) ) ][1]
		nfunctionTodas.append(functionActivationi)
		print("function", functionActivationi )

	print('nNeuronsTodas',nNeuronsTodas)
	if (totalLayers == 1):
		nNeuronsTodas.append(4)
		nfunctionTodas.append('relu')
	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(nNeuronsTodas[0], kernel_size=3, activation=nfunctionTodas[0], input_shape=(100,6,1)))
	model.add(Conv2D(nNeuronsTodas[1], kernel_size=3, activation=nfunctionTodas[1]))
	model.add(Flatten())
	model.add(Dense(3, activation='softmax'))

	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	#train the model
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)

	#predict first 4 images in the test set
	y_pred = model.predict(X_test)
	#actual results for first 4 images in test set
	#print(Y_test[:4])

	test_eval = model.evaluate(X_test, Y_test, verbose=0)
	print('Test loss:', test_eval[0])
	print('Test accuracy:', test_eval[1])
		# Matriz de confusion
	print("""
		Graph confusion matrix...
		""")
	#tabla=pd.crosstab(Y_test.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
	#print(tabla*100/len(y_pred))
	from sklearn.metrics import confusion_matrix
	print('Confusion Matrix')
	print('Y_test',np.argmax(Y_test, axis=1))
	print('y_pred', np.argmax(y_pred, axis=1) )
	confusion = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1))
	print('confusion',confusion*100/len(y_pred))

	predicted_classes = model.predict(X_test)
	predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
	print(predicted_classes.shape, Y_test.shape)
	resultsToReport['X_train']=X_train
	resultsToReport['X_test']=X_test
	resultsToReport['Y_train']=Y_train
	resultsToReport['Y_test']=Y_test
	resultsToReport['y_pred']=y_pred
	resultsToReport['precision'] = test_eval[1]
	resultsToReport['xtrainShape']=xtrainShape
	resultsToReport['ytrainShape']=ytrainShape
	resultsToReport['arregloCuentas'] = arregloCuentas
	resultsToReport['Condusionmatrix'] = confusion*100/len(y_pred)
	resultsToReport['model'] = model

	return resultsToReport


def algorithNeuralNetwork(configuration, X, Y, resultsToReport):
	# neural network
	# configuration {'% Train': '0', 'Total layers': 1, 'Layers': {'Layer 1': ['0', 'Sigmoidal']}}
	print("pAlgorithm 1: Neural Network")
	percentageTrain = configuration['% Train']
	percentageTest = np.around(1-(int(percentageTrain)/100), decimals = 3)


	X, Y = shuffle(X[0:len(Y),:],Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=percentageTest, random_state=0)
	print('There are', X_train.shape[0], 'training data and',  X_test.shape[0], 'testing data.')
	print(np.vstack((np.unique(Y_train), np.bincount(Y_train))).T)

	scaler = StandardScaler() 
	scaler.fit(X_train) 
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)  

	totalLayers = configuration['Total layers']
	infoLayers = configuration['Layers']
	nNeuronsTodas = []
	for i in range(1,totalLayers+1):
		print("layer ", i)
		nNeuronsi = infoLayers[ str('Layer ' + str(i) ) ][0]
		print("neuronas", nNeuronsi )
		nNeuronsTodas.append(int(nNeuronsi))
		functionActivationi = infoLayers[ str('Layer ' + str(i) ) ][1]
		print("function", functionActivationi )

	print("nNeuronsTodas",nNeuronsTodas)
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=nNeuronsTodas, random_state=1)
	clf.fit(X, Y)   

	#####################################################
	# Validation data prediction
	print("""
		Validation data prediction...""")
	y_pred = clf.predict(X_test)
	precision=accuracy_score(Y_test, y_pred)
	print("%.4f" %precision)

	# Matriz de confusion
	print("""
		Graph confusion matrix...
		""")
	tabla=pd.crosstab(Y_test.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
	print(tabla*100/len(y_pred))

	resultsToReport['X_train']=X_train
	resultsToReport['X_test']=X_test
	resultsToReport['Y_train']=Y_train
	resultsToReport['Y_test']=Y_test
	resultsToReport['y_pred']=y_pred
	resultsToReport['precision'] = precision
	resultsToReport['Condusionmatrix'] = tabla*100/len(y_pred)
	resultsToReport['modelMLP'] = clf

	return resultsToReport

# ------------------------------------------------------------- 
#  ____              _      _         ____       _           
# |  _ \  __ _ _ __ (_) ___| | __ _  |  _ \ ___ (_) __ _ ___ 
# | | | |/ _` | '_ \| |/ _ \ |/ _` | | |_) / _ \| |/ _` / __|
# | |_| | (_| | | | | |  __/ | (_| | |  _ < (_) | | (_| \__ \
# |____/ \__,_|_| |_|_|\___|_|\__,_| |_| \_\___// |\__,_|___/
#                                             |__/           
# ------------------------------------------------------------- 




