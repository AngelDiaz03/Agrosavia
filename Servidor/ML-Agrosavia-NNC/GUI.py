# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                                
#     __  _____       ___    ____  ____              ___   ____ ___   ____ 
#    /  |/  / /      /   |  / __ \/ __ \            |__ \ / __ \__ \ / __ \
#   / /|_/ / /      / /| | / /_/ / /_/ /  ______    __/ // / / /_/ // / / /
#  / /  / / /___   / ___ |/ ____/ ____/  /_____/   / __// /_/ / __// /_/ / 
# /_/  /_/_____/  /_/  |_/_/   /_/                /____/\____/____/\____/  
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                                
# GUI.py 
# Author: Daniela Rojas
# A graphical user interface to train machine learning models.
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                               
    



import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Resources.modelTraining import trainModel
from Resources.reportGeneration import report
from Resources.Dialogs import *
import platform
import logging
from datetime import datetime
import os
logging.basicConfig(filename='log.log', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logging.info("Machine Learning - Model Training started")

minTkVersion = 8.6

if tk.TkVersion < minTkVersion:
	logging.warning("UPDATE YOUR PYTHON VERSION (DEPRECATED TK VERSION), GUI MAY NOT WORK AS EXPECTED")
	print("UPDATE YOUR PYTHON VERSION (DEPRECATED TK VERSION), GUI MAY NOT WORK AS EXPECTED")

class MainGUI:

	def __init__(self):

		self.root = tk.Tk()
		self.root.configure(background='white')
		self.root.title("Machine Learning - Model Training")
		self.img_icon = ImageTk.PhotoImage(Image.open('./Resources/icon.png'))
		self.root.tk.call('wm', "iconphoto", self.root._w, self.img_icon)
		self.df = None
		self.root.filename = ""
		self.dataFrames = []
		self.dataProcessing = 999
		self.algorithm = 999
		self.configuration = {}
		self.model = None
		self.folder = None
		self.optionsDataProcessing = ["Use all data available"]
		self.optionsAlgorithm = ["Convolutional Neural Network"]

		self.imgLogo = ImageTk.PhotoImage(Image.open("./Resources/logo.jpg").resize((209, 50), Image.BICUBIC))
		self.logo = Label(self.root, image = self.imgLogo, borderwidth=0).pack(side=TOP, padx=10, pady=(25,10), expand=YES)



		self.frame = Frame(self.root, background='white')
		Label(self.frame, text = "File:  ", background='white', borderwidth=0).pack(side=LEFT)
		self.filename_Entry = tk.Entry(self.frame)
		self.filename_Entry.bindtags((str(self.filename_Entry), str(self.frame), "all"))
		self.filename_Entry.pack(side=LEFT, fill=BOTH, expand=YES)
		self.imgSeach = ImageTk.PhotoImage(Image.open("./Resources/search.png").resize((20, 20), Image.BICUBIC))
		self.imgSearchButton = Button(self.frame, command = self.openFileChooser, image = self.imgSeach)
		self.imgSearchButton.pack(side=LEFT)
		self.frame.pack(fill=BOTH, expand=True, pady=(30,0),padx=(18,25))

		

		self.processButton = Button(self.root, command=self.processData, text = "Process data from file", width=24)
		self.processButton.pack(pady=30, expand=YES)


		self.terminalFont = "Monaco" if platform.system() == "Darwin" else "Monospace"
		self.terminalSize = 11 if platform.system() == "Darwin" else 9
		self.console = tk.Text(self.root, height=9, width=48, font=(self.terminalFont, self.terminalSize), background='black', foreground='white', highlightthickness=0)
		self.console.insert(END, "Welcome to ML - Model Training")
		self.console.bind("<Key>", self.disableWriting)
		self.console.pack(padx=10, pady=10, expand=YES)


		self.infoCR = Button(self.root,text="© Info Copyright ",command=self.openInfoCR )
		self.infoCR.pack(side=RIGHT, pady=0)


		self.root.resizable(False, False)
		self.root.wait_visibility()
		self.root.withdraw()
		center(self.root, windowWidth=self.root.winfo_reqwidth(), windowHeight=self.root.winfo_reqheight())
		self.root.deiconify()
		self.root.mainloop()

	def openInfoCR(self):
		CR_dialog = infoDialog(self.root, "© Info Copyright")
		self.root.wait_window(CR_dialog.top)

	def openFileChooser(self):
		self.root.filename = filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("CSV files","*.csv"),("Excel files","*.xls *.xlsx")))
		if self.root.filename:
			self.logConsole("Selected file: "+self.root.filename.split("/")[-1])
			self.filename_Entry.delete(0, tk.END)
			self.filename_Entry.insert(0, self.root.filename.split("/")[-2]+"/"+self.root.filename.split("/")[-1])

	def processData(self):
		self.processButton.config(state=DISABLED)
		self.imgSearchButton.config(state=DISABLED)
		self.logConsole("---------------------------")
		if self.root.filename.endswith("csv"):
			self.logConsole("Processing "+self.root.filename.split("/")[-1])
			self.root.update()
			self.df = pd.read_csv(self.root.filename, header=None)
		elif self.root.filename.endswith("xls") or self.root.filename.endswith("xlsx"):
			self.logConsole("Processing "+self.root.filename.split("/")[-1])
			self.root.update()
			self.df = pd.read_excel(self.root.filename, header=None)
		else:
			self.logConsole("Wrong file format", error=True)
			self.processButton.config(state=NORMAL)
			self.imgSearchButton.config(state=NORMAL)
			return
		try:
			self.logConsole("Counting and sorting module data")
			self.root.update()
			if self.df.shape[1]%10 != 0:
				raise Exception("Data format error")

			self.dataFrames = []
			for module in range(int(self.df.shape[1]/10)):
				actualDF = self.df[[(x+10*module) for x in range(10)]]
				actualDF.columns = ['date','module','packet', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'action']
				self.dataFrames.append(actualDF)

			self.logConsole("Dropping empty rows")
			for x in range(len(self.dataFrames)):
				self.dataFrames[x] = self.dataFrames[x].dropna(axis=0, how='all')
				self.dataFrames[x] = self.dataFrames[x].reset_index(drop=True)
			
			self.logConsole("Modules detected: "+str(len(self.dataFrames)))
			for x in range(len(self.dataFrames)):
				self.logConsole("Rows in module "+str(x+1)+": "+str(self.dataFrames[x].shape[0]))
			

			d_DP = optionDialog(self.root, "What kind of data processing should I use?", self.optionsDataProcessing)
			self.root.wait_window(d_DP.top)
			self.dataProcessing = d_DP.getValue()

			if self.dataProcessing == -1:
				self.logConsole("Processing cancelled")
				self.processButton.config(state=NORMAL)
				self.imgSearchButton.config(state=NORMAL)
				return

			d_ML = optionDialog(self.root, "Select a Machine Learning algorithm", self.optionsAlgorithm)
			self.root.wait_window(d_ML.top)
			self.algorithm = d_ML.getValue()

			if self.algorithm == -1:
				self.logConsole("Processing cancelled")
				self.processButton.config(state=NORMAL)
				self.imgSearchButton.config(state=NORMAL)
				return

			#Configuration windows
			# if self.algorithm == 0: #Random Forest config
			# 	RF_config = randomForestConfigDialog(self.root, len(self.dataFrames))
			# 	self.root.wait_window(RF_config.top)
			# 	self.configuration = RF_config.getConfiguration()
			# elif self.algorithm == 1: #Neural Network config
			# 	NN_config = neuralNetworkConfigDialog(self.root, len(self.dataFrames))
			# 	self.root.wait_window(NN_config.top)
			# 	self.configuration = NN_config.getConfiguration()
			if self.algorithm == 0: #Convolutional Neural Network config
				CNN_config = convolutionNeuralNetworkConfigDialog(self.root, len(self.dataFrames))
				self.root.wait_window(CNN_config.top)
				self.configuration = CNN_config.getConfiguration()
			else:
				self.configuration = {}
				raise Exception("Internal Error")

			if self.configuration == -1:
				self.logConsole("Processing cancelled")
				self.processButton.config(state=NORMAL)
				self.imgSearchButton.config(state=NORMAL)
				return

			self.logConsole("Data processing: " + self.optionsDataProcessing[self.dataProcessing])
			self.logConsole("Algorithm: " + self.optionsAlgorithm[self.algorithm])

			for key, value in self.configuration.items():
				if key == "Layers":
					for keyL, valueL in value.items():
						self.logConsole( keyL+" -> Neurons: "+str(valueL[0]+"  Act. Fn: "+valueL[1]))
				else:
					self.logConsole( key+": "+str(value).strip("[").strip("]") )

			self.logConsole("TRAINING...")
			self.root.update()
			
			self.model = trainModel(self.dataFrames, self.dataProcessing, self.algorithm, self.configuration)
			self.logConsole("Training completed successfully!")

			self.logConsole("Generating plots and model analysis")
			self.root.update()

			self.folder = plotAndAnalysis(self)
			self.logConsole("Plots and analysis generated")
			self.logConsole("Folder: "+self.folder)

			self.processButton.config(state=NORMAL)
			self.imgSearchButton.config(state=NORMAL)


		except Exception as e:
			self.logConsole("Critical ERROR in data", error = True)
			self.processButton.config(state=NORMAL)
			self.imgSearchButton.config(state=NORMAL)
			print("ERROR: ",str(e))
			raise e

	def logConsole(self, text, error = False):
		self.console.insert(END, "\n"+text)
		self.console.see(END)
		if not error:
			logging.info(text)
		else:
			logging.error(text)

	def disableWriting(self, event):
		return "break"


def plotAndAnalysis(GUI):

	directory = "Results/"+datetime.now().strftime("%m-%d-%Y, %H %M %S") + " - "+GUI.optionsAlgorithm[GUI.algorithm]
	os.mkdir(directory)

	reportObj = report()
	reportObj.datetime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
	reportObj.platform = platform.platform()
	reportObj.filename = GUI.root.filename
	reportObj.modulesDetected = len(GUI.dataFrames)
	reportObj.dataResults = GUI.model

	reportObj.rowsDetected = []
	for dfDetecter in GUI.dataFrames:
		reportObj.rowsDetected.append(dfDetecter.shape[0])

	reportObj.dataProcessing = GUI.optionsDataProcessing[GUI.dataProcessing]
	reportObj.algorithm = GUI.optionsAlgorithm[GUI.algorithm]
	reportObj.configuration = GUI.configuration

	reportObj.export(directory, "report.txt")
	reportObj.exportPlots(directory)
	reportObj.saveModel(directory)
	return directory
		
mainGUI = MainGUI()


# ------------------------------------------------------------- 
#  ____              _      _         ____       _           
# |  _ \  __ _ _ __ (_) ___| | __ _  |  _ \ ___ (_) __ _ ___ 
# | | | |/ _` | '_ \| |/ _ \ |/ _` | | |_) / _ \| |/ _` / __|
# | |_| | (_| | | | | |  __/ | (_| | |  _ < (_) | | (_| \__ \
# |____/ \__,_|_| |_|_|\___|_|\__,_| |_| \_\___// |\__,_|___/
#                                             |__/           
# ------------------------------------------------------------- 


