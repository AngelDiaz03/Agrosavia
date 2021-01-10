import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas
from sklearn.externals import joblib
import pickle

class report:
	def __init__(self):
		self.datetime = None
		self.platform = None
		self.filename = None
		self.modulesDetected = None
		self.rowsDetected = None
		self.dataProcessing = None
		self.algorithm = None
		self.configuration = None
		self.f = None
		self.dataResults = None

	def export(self, directory, file):

		self.f = open(directory+"/"+file,"w+")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("///////////// Machine Learning - Model Training - Report /////////////////////////////////")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("File original directory: "+directory+"/"+file)
		self.writeLine("Datetime of generation: "+self.datetime)
		self.writeLine("OS of generation: "+self.platform)
		self.writeLine("File processed: "+self.filename)
		self.writeLine("Modules detected in file: "+str(self.modulesDetected))
		for index, row in zip(range(len(self.rowsDetected)), self.rowsDetected):
			self.writeLine("Rows detected in module "+str(index+1)+": "+str(row))
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("//////////////////////// DATA ENTERED BY USER ////////////////////////////////////////////")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("Data processing requested was "+self.dataProcessing)
		self.writeLine("Algorithm requested was "+self.algorithm)

		self.writeLine("CONFIGURATION:")
		for key, value in self.configuration.items():
				if key == "Layers":
					for keyL, valueL in value.items():
						self.writeLine( keyL+" -> Neurons: "+str(valueL[0]+"  Activation Function: "+valueL[1]))
				else:
					self.writeLine( key+": "+str(value).strip("[").strip("]") )

		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("////////////////////// Features of Input Data ////////////////////////////////////////////")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")

		self.writeLine("DATA READY: ")
		self.writeLine("All info of activates modules read, labels was converted to numbers...")
		self.writeLine(" CLASS = ACTION ")
		self.writeLine("   0   = Comiendo ")
		self.writeLine("   1   = Rumiando " )
		self.writeLine("   2   = Otro")
		self.writeLine(" ")
		self.writeLine(" X Shape: " + str(self.dataResults['X'].shape))
		self.writeLine(" Y Shape: " + str(self.dataResults['Y'].shape))
		self.writeLine("  ")
		
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("////////////////////// Results ML Model //////////////////////////////////////////////////")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")


		if self.algorithm == "Random Forest":
			self.writeLine("Random Forest:")
			if self.configuration['Hyperparameter optimization']==True:
				for trainingN in range(1,int(self.configuration['# of Trainings'])+1):
					self.writeLine("------------------------------------------------------------------------------------------")
					self.writeLine("Training number "+ str( trainingN))
					self.writeLine('There are ' + str(self.dataResults['X_train_'+str(trainingN)].shape[0]) + ' training data and ' + str(self.dataResults['X_test_'+str(trainingN)].shape[0])+ ' testing data. ')
					self.writeLine(np.array2string(np.vstack((np.unique(self.dataResults['Y_train_'+str(trainingN)]), np.bincount(self.dataResults['Y_train_'+str(trainingN)]))).T ))
					self.writeLine(" ")
					self.writeLine("After looking for parameters . . . ")
					self.writeLine("The best parameters found by the optimization method was:")
					self.writeLine("   n_estimators       =   "+str(self.dataResults["Opti_nEstimators_"+str(trainingN)]))
					self.writeLine("   max_features       =   "+str(self.dataResults["Opti_max_feature_"+str(trainingN)]))					
					self.writeLine("   n_samples_leaf     =   "+str(self.dataResults["Opti_samples_leaf_"+str(trainingN)]))
					self.writeLine("And the accuaracy precission with this parameters was:")					
					self.writeLine("   Accuaracy precission Percentage    =   "+str(int((self.dataResults['OptimaxPrecision_'+str(trainingN)])*100)))

			else:
				for trainingN in range(1,int(self.configuration['# of Trainings'])+1):
					self.writeLine("------------------------------------------------------------------------------------------")
					self.writeLine("Training number "+ str( trainingN))
					self.writeLine('There are ' + str(self.dataResults['X_train_'+str(trainingN)].shape[0]) + ' training data and ' + str(self.dataResults['X_test_'+str(trainingN)].shape[0])+ ' testing data. ')
					self.writeLine(np.array2string(np.vstack((np.unique(self.dataResults['Y_train_'+str(trainingN)]), np.bincount(self.dataResults['Y_train_'+str(trainingN)]))).T ))
					self.writeLine(" ")
					self.writeLine("After training random forest . . .")
					self.writeLine('The Calculated importance of variables for prediction was:')
					self.writeLine("   "+ str(np.around(self.dataResults['importanciaVars_'+str(trainingN)], decimals = 3)))
					self.writeLine(" ")
					self.writeLine("""Validation data prediction was: """ + str("%.4f" % self.dataResults['precision_'+str(trainingN)]))
					self.writeLine(" ")
					self.writeLine(""" Graph confusion matrix... """)
					self.writeLine(np.array2string(np.array(self.dataResults['Condusionmatrix_'+str(trainingN)] ) ))
		elif self.algorithm == "Neural Network":
			self.writeLine("Neural Network")
			self.writeLine("------------------------------------------------------------------------------------------")
			self.writeLine('There are ' + str(self.dataResults['X_train'].shape[0]) + ' training data and ' + str(self.dataResults['X_test'].shape[0])+ ' testing data. ')
			self.writeLine(np.array2string(np.vstack((np.unique(self.dataResults['Y_train']), np.bincount(self.dataResults['Y_train']))).T ))
			self.writeLine(" ")
			self.writeLine("After training neural Network . . .")
			self.writeLine("""Validation data prediction was: """ + str("%.4f" % self.dataResults['precision']))
			self.writeLine(" ")
			self.writeLine(""" Graph confusion matrix... """)
			self.writeLine(np.array2string(np.array(self.dataResults['Condusionmatrix'] ) ))

		elif self.algorithm == "Convolutional Neural Network":
			self.writeLine("Convolutional Neural Network")
			self.writeLine("------------------------------------------------------------------------------------------")
			self.writeLine('There are ' + str(self.dataResults['xtrainShape']) + ' training data and ' + str(self.dataResults['ytrainShape'])+ ' testing data. ')
			self.writeLine(np.array2string((self.dataResults['arregloCuentas'])))
			self.writeLine(" ")
			self.writeLine("After training neural Network . . .")
			self.writeLine("""Validation data prediction was: """ + str("%.4f" % self.dataResults['precision']))
			self.writeLine(" ")
			self.writeLine(""" Graph confusion matrix... """)
			self.writeLine(np.array2string(np.array(self.dataResults['Condusionmatrix'] ) ))
	
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.writeLine("/////////////////////////// END OF REPORT ////////////////////////////////////////////////")
		self.writeLine("//////////////////////////////////////////////////////////////////////////////////////////")
		self.f.close()


	def saveModel(self,directory):

		if self.algorithm == "Random Forest":
			if self.configuration['Hyperparameter optimization']==True:
				for trainingN in range(1,int(self.configuration['# of Trainings'])+1):
					fileName = "model"+str(trainingN)
					model = self.dataResults['OptiModel_'+str(trainingN)]
					joblib.dump(model, directory+"/"+fileName+'.csv')
					with open(fileName+'.pkl', 'wb') as f:
						pickle.dump(model, f)

			else:
				for trainingN in range(1,int(self.configuration['# of Trainings'])+1):
					fileName = "model"+str(trainingN)
					model = self.dataResults['modelRF'+str(trainingN)]
					joblib.dump(model, directory+"/"+fileName+'.csv')
					with open(fileName+'.pkl', 'wb') as f:
						pickle.dump(model, f)

		elif self.algorithm == "Neural Network":
			fileName = "model"
			model = self.dataResults['modelMLP']
			joblib.dump(model, directory+"/"+fileName+'.csv')
			s = pickle.dumps(model)

		elif self.algorithm == "Convolutional Neural Network":
			model = self.dataResults['model']
			model.save(directory+"/"+"model.h5py")
			# serialize model to JSON
			model_json = model.to_json()
			with open(directory+"/"+"model.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model.save_weights(directory+"/"+"model.h5")
			print("Saved model to disk")

	def exportPlots(self,directory):
		### Feature Vs Feature
		plt.rcParams["figure.figsize"] = (12,7)

		fileName = "Feature X1 Vs Feature X6"
		f, ax1 = plt.subplots()
		f.suptitle('Feature X1 Vs Feature X6', fontsize=20)
		ax1.set_xlabel('$x_1$' , fontsize=18)
		ax1.set_ylabel('$x_6$' , fontsize=18)
		ax1.minorticks_on()
		ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

		volume =  self.dataResults['X'][:,0]
		amount =  self.dataResults['X'][:,4]
		ranking = self.dataResults['Y']
		scatter = ax1.scatter(volume, amount, c=ranking, s=40)

		#plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)


		if self.algorithm == "Random Forest":
			if self.configuration['Hyperparameter optimization']==True:
				for trainingN in range(1,int(self.configuration['# of Trainings'])+1):
					### Y train Class Distribution 
					fileName = "Y_train_Class_Distribution_"+str(trainingN)
					f, ax1 = plt.subplots()
					f.suptitle('Y_train Class Distribution, for training '+str(trainingN), fontsize=20)
					ax1.set_xlabel('Class' , fontsize=18)
					ax1.set_ylabel('Amount' , fontsize=18)				
					sns.countplot(self.dataResults['Y_train_'+str(trainingN)])

					plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

				### plot parameters optimizating
				fileName = "n_estimators_opti"
				f, ax1 = plt.subplots()
				f.suptitle('Optimization n_estimators', fontsize=20)
				ax1.set_xlabel('Number of estimators' , fontsize=18)
				ax1.set_ylabel('Accuaracy percentage' , fontsize=18)

				ax1.scatter(self.dataResults['resultadosAllCasesOpti_'+str(1)][1],self.dataResults['resultadosAllCasesOpti_'+str(1)][4] , c='red', label='n_estimators'+str(1)   )
				ax1.legend(bbox_to_anchor=(0.78, 0.95), loc=2, borderaxespad=0.)

				plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)



			else:
				for trainingN in range(1,int(self.configuration['# of Trainings'])+1):
					### Y train Class Distribution 
					fileName = "Y_train_Class_Distribution_"+str(trainingN)
					f, ax1 = plt.subplots()
					f.suptitle('Y_train Class Distribution, for training '+str(trainingN), fontsize=20)
					ax1.set_xlabel('Class' , fontsize=18)
					ax1.set_ylabel('Amount' , fontsize=18)
					sns.countplot(self.dataResults['Y_train_'+str(trainingN)])

					plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

					### Plot Importance Varaibles
					fileName = "variable_importance"+str(trainingN)
					f, ax1 = plt.subplots()
					f.suptitle('Variable importance, for training '+str(trainingN), fontsize=20)
					ax1.set_ylabel('Feature' , fontsize=18)
					ax1.set_xlabel('Variable importance percentage' , fontsize=18)
					pos=[1, 2, 3, 4, 5, 6]
					ax1.set_yticks(pos)
					ax1.set_yticklabels([ "X1" , "X2", "X3" , "X4", "X5", "X6"])
					ax1.invert_yaxis()  # labels read top-to-bottom
					ax1.barh(pos, self.dataResults['importanciaVars_'+str(trainingN)], align='center',color='blue')

					plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)
		elif self.algorithm == "Neural Network":
			fileName = "Y_train_Class_Distribution_"
			f, ax1 = plt.subplots()
			f.suptitle('Y_train Class Distribution ', fontsize=20)
			ax1.set_xlabel('Class' , fontsize=18)
			ax1.set_ylabel('Amount' , fontsize=18)				
			sns.countplot(self.dataResults['Y_train'])

			plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)
		elif self.algorithm == "Convolutional Neural Network":
			fileName = "Y_train_Class_Distribution_"
			# f, ax1 = plt.subplots()
			# f.suptitle('Y_train Class Distribution ', fontsize=20)
			# ax1.set_xlabel('Class' , fontsize=18)
			# ax1.set_ylabel('Amount' , fontsize=18)				
			# sns.countplot(self.dataResults['Y_train'])

			# plt.savefig(directory+"/"+fileName+'.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)
			
	def writeLine(self, text):
		self.f.write(text+"\n")
