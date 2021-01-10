# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                                
#     __  _____       ___    ____  ____              ___   ____ ___   ____ 
#    /  |/  / /      /   |  / __ \/ __ \            |__ \ / __ \__ \ / __ \
#   / /|_/ / /      / /| | / /_/ / /_/ /  ______    __/ // / / /_/ // / / /
#  / /  / / /___   / ___ |/ ____/ ____/  /_____/   / __// /_/ / __// /_/ / 
# /_/  /_/_____/  /_/  |_/_/   /_/                /____/\____/____/\____/  
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                                
# Dialogs.py 
# Author: Daniela Rojas 
# Configuration and setup of each dialog
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––                                                               
   
from tkinter import *
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import ttk
import platform

def center(win, windowWidth, windowHeight):
		positionRight = int(win.winfo_screenwidth()/2 - windowWidth/2)
		positionDown = int(win.winfo_screenheight()/2 - windowHeight/2)
		win.geometry("+{}+{}".format(positionRight, positionDown))

class infoDialog:

	def __init__(self, parent, text):

		top = self.top = Toplevel(parent)
		self.top.title("Info")
		self.top.resizable(False, False)
		self.img_icon = ImageTk.PhotoImage(Image.open('./Resources/icon.png'))
		self.top.tk.call('wm', "iconphoto", self.top._w, self.img_icon)
		self.selectedOk = False

		Label(top, text=text).pack(pady=15, padx=7)
		textoinfo = " Author: \n\n Daniela Rojas Lozano \n ddp.rojas11@uniandes.edu.co \n\n\n Adviser: \n\n Fredy Enrique Segura \n fsegura@uniandes.edu.co\n\n\n  Departamento de Ingeniería Eléctrica y Electrónica \n Universidad de los Andes"
		self.infoCR = Label(top,text=textoinfo)
		self.infoCR.pack()

		b = Button(top, text="Continue", command=self.ok, width=10)
		b.pack(pady=15, side=BOTTOM)

		top.wait_visibility()
		top.withdraw()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		top.deiconify()
		top.grab_set()

	def ok(self):
		self.selectedOk = True
		self.top.destroy()


class optionDialog:

	def __init__(self, parent, text, options):

		top = self.top = Toplevel(parent)
		self.top.title("Select option")
		self.top.resizable(False, False)
		self.img_icon = ImageTk.PhotoImage(Image.open('./Resources/icon.png'))
		self.top.tk.call('wm', "iconphoto", self.top._w, self.img_icon)
		self.selectedOk = False

		Label(top, text=text).pack(pady=15, padx=7)

		self.v = tk.IntVar()
		self.v.set(0)
		for val, option in enumerate(options):
			tk.Radiobutton(top, text=option, indicatoron = 0, width = 22, padx = 20, variable=self.v, value=val, selectcolor='gray25').pack()

		b = Button(top, text="Continue", command=self.ok, width=10)
		b.pack(pady=15, side=BOTTOM)

		top.wait_visibility()
		top.withdraw()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		top.deiconify()
		top.grab_set()

	def ok(self):
		self.selectedOk = True
		self.top.destroy()

	def getValue(self):
		if self.selectedOk:
			return self.v.get()
		else:
			return -1

	

class randomForestConfigDialog:

	def __init__(self, parent, nModules):

		top = self.top = Toplevel(parent)
		self.top.title("Random Forest configuration")
		self.top.resizable(False, False)
		self.img_icon = ImageTk.PhotoImage(Image.open('./Resources/icon.png'))
		self.top.tk.call('wm', "iconphoto", self.top._w, self.img_icon)
		self.selectedOk = False
		self.hyperparameters = 999
		self.moduleList = []
		self.configuration = {}

		Label(top, text="Random Forest configuration", font=("", 15)).pack(pady=(0,10))

		
		self.fields = ["% Train", "# of Trainings", "Hyperparameter optimization", "N-estimators", "Max features", "Min samples leaf"]
		self.types = ["entry", "entry", "enable", "entry", "entry", "entry"]
		self.entries = {}
		for field, type_f in zip(self.fields, self.types):
			row = Frame(self.top)
			lab = Label(row, width=29, text=field+": ", anchor='w')
			if type_f == "enable":
				self.hyperparameters = IntVar()
				ent = Checkbutton(row, variable=self.hyperparameters, command=self.cb)
			else:
				ent = Entry(row)
				ent.insert(0,"0")
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
			lab.pack(side = LEFT)
			ent.pack(side = RIGHT, expand = YES, fill = X)
			self.entries[field] = self.hyperparameters if type_f == "enable" else ent

		Label(self.top, anchor='w', text="Modules to use: ").pack(side = TOP, fill=X, padx = 5 , pady = 5)
		nextRow = True
		for x in range(nModules):
			if nextRow:
				row = Frame(self.top)
				nextRow = False

			var = IntVar(value=1)
			ent = Checkbutton(row, variable=var, text="Module "+str(x+1))
			self.moduleList.append(var)

			if x%2 == 1:
				nextRow = True
				ent.pack(side = RIGHT, expand = YES)
				row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
			else:
				ent.pack(side = LEFT, expand = YES)

		if nextRow == False:
			color = top.cget('bg')
			entAux = Checkbutton(row, text="Module AAA", bg=color, fg=color, indicatoron = 0, borderwidth=0, activebackground=color, activeforeground=color, highlightcolor=color, selectcolor=color)
			entAux.pack(side = RIGHT, expand = YES)
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)



		b = Button(top, text="Start training", command=self.ok, width=15)
		b.pack(pady=15, side=BOTTOM)

		top.wait_visibility()
		top.withdraw()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		top.deiconify()
		top.grab_set()

	def ok(self):
		self.selectedOk = True

		for field, type_f in zip(self.fields, self.types):
			self.configuration[field] = self.entries[field].get() if type_f == "entry" else self.hyperparameters.get() == 1

		self.configuration["Active Module IDs"] = []
		for index, moduleVar in zip(range(len(self.moduleList)), self.moduleList):
			if moduleVar.get() == 1:
				self.configuration["Active Module IDs"].append(index+1)

		if self.hyperparameters.get() == 1:
			del self.configuration["N-estimators"]
			del self.configuration["Max features"]
			del self.configuration["Min samples leaf"]

		self.top.destroy()

	def cb(self):
		if self.hyperparameters.get() == 1:
			self.entries["N-estimators"].config(state=DISABLED)
			self.entries["Max features"].config(state=DISABLED)
			self.entries["Min samples leaf"].config(state=DISABLED)
		else:
			self.entries["N-estimators"].config(state=NORMAL)
			self.entries["Max features"].config(state=NORMAL)
			self.entries["Min samples leaf"].config(state=NORMAL)

	def getConfiguration(self):
		if self.selectedOk:
			return self.configuration
		else:
			return -1

class convolutionNeuralNetworkConfigDialog:
	def __init__(self, parent, nModules):

		top = self.top = Toplevel(parent)
		self.top.title("Neural Network configuration")
		self.top.resizable(False, False)
		self.img_icon = ImageTk.PhotoImage(Image.open('./Resources/icon.png'))
		self.top.tk.call('wm', "iconphoto", self.top._w, self.img_icon)
		self.selectedOk = False
		self.moduleList = []
		self.numLayers = 1
		self.configuration = {}
		self.activationFunctions = ["softmax", "relu"]
		self.rowsForget = []
		self.neuronsEntry = []
		self.activationFunctionsEntry = []

		Label(top, text="Neural Network configuration", font=("", 15)).pack(pady=(0,10))

		self.fields = ["% Train"]
		self.entries = {}
		for field in self.fields:
			row = Frame(self.top)
			lab = Label(row, width=15, text=field+": ", anchor='w')
			ent = Entry(row)
			ent.insert(0,"0")
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
			lab.pack(side = LEFT)
			ent.pack(side = RIGHT, expand = YES, fill = X)
			self.entries[field] = ent

		#Layers logic
		self.frameLayers = Frame(self.top)

		style = ttk.Style()

		style.map('TCombobox', fieldbackground=[('readonly','white')])
		style.map('TCombobox', selectbackground=[('readonly', 'white')])
		style.map('TCombobox', background=[('readonly', 'white')])
		style.map('TCombobox', selectforeground=[('readonly', 'black')])
 
	
		numberLayersFrame = Frame(self.frameLayers)
		lab = Label(numberLayersFrame, width=15, text="Number of layers: ", anchor='w' )
		self.dropdown_value = StringVar(value="1")
		self.dropdown = ttk.Combobox(numberLayersFrame, values=list(range(1,3)), state="readonly", style="TCombobox", textvariable = self.dropdown_value)
		self.dropdown.bind("<<ComboboxSelected>>", self.callbackNumLayers)
		numberLayersFrame.pack(side = TOP, fill = X, pady=(0,10))
		lab.pack(side = LEFT)
		self.dropdown.pack(side = RIGHT, expand = YES, fill = X)

		self.updateLayerFields()
		self.frameLayers.pack(side = TOP, fill = X, padx = 5 , pady = 5)


		Label(self.top, anchor='w', text="Modules to use: ").pack(side = TOP, fill=X, padx = 5 , pady = 5)
		nextRow = True
		for x in range(nModules):
			if nextRow:
				row = Frame(self.top)
				nextRow = False

			var = IntVar(value=1)
			ent = Checkbutton(row, variable=var, text="Module "+str(x+1))
			self.moduleList.append(var)

			if x%2 == 1:
				nextRow = True
				ent.pack(side = RIGHT, expand = YES)
				row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
			else:
				ent.pack(side = LEFT, expand = YES)

		if nextRow == False:
			color = top.cget('bg')
			entAux = Checkbutton(row, text="Module AAA", bg=color, fg=color, indicatoron = 0, borderwidth=0, activebackground=color, activeforeground=color, highlightcolor=color, selectcolor=color)
			entAux.pack(side = RIGHT, expand = YES)
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)

		b = Button(top, text="Start training", command=self.ok, width=15)
		b.pack(pady=15, side=BOTTOM)

		top.wait_visibility()
		top.withdraw()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		top.deiconify()
		top.grab_set()

	def updateLayerFields(self):

		for rowF in self.rowsForget:
			rowF.pack_forget()
			self.rowsForget = []

		self.neuronsEntry = []
		self.activationFunctionsEntry = []

		nextRow = True
		for fieldGroupID in range(self.numLayers):
			if nextRow:
				row = Frame(self.frameLayers)
				self.rowsForget.append(row)
				nextRow = False

			rowInner = LabelFrame(row, text = "Layer "+str(fieldGroupID+1))

			row1 = Frame(rowInner)
			widthNeurons = 15 if platform.system() == "Darwin" else 18
			lab = Label(row1, width=widthNeurons, text="Neurons Convolution:", anchor='w')
			ent = Entry(row1, width=12) #Entry Neurons
			self.neuronsEntry.append(ent)
			ent.insert(0,"0")
			lab.pack(side = LEFT, pady=(5,0), padx=(5,0))
			ent.pack(side = RIGHT, pady=(5,0), padx=(0,5), expand = YES, fill=X)

			row2 = Frame(rowInner)
			lab2 = Label(row2, width=15, text="Act. Function:", anchor='w')
			optionActVar = StringVar(value=self.activationFunctions[0])
			ent2 = OptionMenu(row2, optionActVar, *self.activationFunctions) #Entry Act Function
			self.activationFunctionsEntry.append(optionActVar)
			ent2.config(width=10)
			lab2.pack(side = LEFT, pady=(0,5), padx=(5,0))
			ent2.pack(side = RIGHT, pady=(0,5), padx=(0,5))

			row1.pack(side = TOP)
			row2.pack(side = TOP)

			expand = YES if fieldGroupID==0 else NO
			rowInner.pack(side = LEFT, fill = X, padx = 5 , pady = 5, expand = expand)

			
			if fieldGroupID%3 == 2:
				nextRow = True
				row.pack(side = TOP, fill = X, padx = 5 , pady = 5)


		if nextRow == False:
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)

	def callbackNumLayers(self, event):
		self.numLayers = int(self.dropdown.get())
		self.updateLayerFields()
		self.top.withdraw()
		self.top.update_idletasks()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		self.top.deiconify()

	def ok(self):
		self.selectedOk = True

		for field in self.fields:
			self.configuration[field] = self.entries[field].get()
		self.configuration["Total layers"] = self.numLayers

		self.configuration["Layers"] = {}
		for index, neuron, act_fn in zip(range(len(self.neuronsEntry)), self.neuronsEntry, self.activationFunctionsEntry):
			self.configuration["Layers"]["Layer "+str(index+1)] = [neuron.get(), act_fn.get()]


		self.configuration["Active Module IDs"] = []
		for index, moduleVar in zip(range(len(self.moduleList)), self.moduleList):
			if moduleVar.get() == 1:
				self.configuration["Active Module IDs"].append(index+1)

		self.top.destroy()

	def getConfiguration(self):
		if self.selectedOk:
			return self.configuration
		else:
			return -1



class neuralNetworkConfigDialog:

	def __init__(self, parent, nModules):

		top = self.top = Toplevel(parent)
		self.top.title("Neural Network configuration")
		self.top.resizable(False, False)
		self.img_icon = ImageTk.PhotoImage(Image.open('./Resources/icon.png'))
		self.top.tk.call('wm', "iconphoto", self.top._w, self.img_icon)
		self.selectedOk = False
		self.moduleList = []
		self.numLayers = 1
		self.configuration = {}
		self.activationFunctions = ["Sigmoidal", "Linear"]
		self.rowsForget = []
		self.neuronsEntry = []
		self.activationFunctionsEntry = []

		Label(top, text="Neural Network configuration", font=("", 15)).pack(pady=(0,10))

		self.fields = ["% Train"]
		self.entries = {}
		for field in self.fields:
			row = Frame(self.top)
			lab = Label(row, width=15, text=field+": ", anchor='w')
			ent = Entry(row)
			ent.insert(0,"0")
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
			lab.pack(side = LEFT)
			ent.pack(side = RIGHT, expand = YES, fill = X)
			self.entries[field] = ent

		#Layers logic
		self.frameLayers = Frame(self.top)

		style = ttk.Style()

		style.map('TCombobox', fieldbackground=[('readonly','white')])
		style.map('TCombobox', selectbackground=[('readonly', 'white')])
		style.map('TCombobox', background=[('readonly', 'white')])
		style.map('TCombobox', selectforeground=[('readonly', 'black')])
 
	
		numberLayersFrame = Frame(self.frameLayers)
		lab = Label(numberLayersFrame, width=15, text="Number of layers: ", anchor='w' )
		self.dropdown_value = StringVar(value="1")
		self.dropdown = ttk.Combobox(numberLayersFrame, values=list(range(1,16)), state="readonly", style="TCombobox", textvariable = self.dropdown_value)
		self.dropdown.bind("<<ComboboxSelected>>", self.callbackNumLayers)
		numberLayersFrame.pack(side = TOP, fill = X, pady=(0,10))
		lab.pack(side = LEFT)
		self.dropdown.pack(side = RIGHT, expand = YES, fill = X)

		self.updateLayerFields()
		self.frameLayers.pack(side = TOP, fill = X, padx = 5 , pady = 5)


		Label(self.top, anchor='w', text="Modules to use: ").pack(side = TOP, fill=X, padx = 5 , pady = 5)
		nextRow = True
		for x in range(nModules):
			if nextRow:
				row = Frame(self.top)
				nextRow = False

			var = IntVar(value=1)
			ent = Checkbutton(row, variable=var, text="Module "+str(x+1))
			self.moduleList.append(var)

			if x%2 == 1:
				nextRow = True
				ent.pack(side = RIGHT, expand = YES)
				row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
			else:
				ent.pack(side = LEFT, expand = YES)

		if nextRow == False:
			color = top.cget('bg')
			entAux = Checkbutton(row, text="Module AAA", bg=color, fg=color, indicatoron = 0, borderwidth=0, activebackground=color, activeforeground=color, highlightcolor=color, selectcolor=color)
			entAux.pack(side = RIGHT, expand = YES)
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)

		b = Button(top, text="Start training", command=self.ok, width=15)
		b.pack(pady=15, side=BOTTOM)

		top.wait_visibility()
		top.withdraw()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		top.deiconify()
		top.grab_set()

	def updateLayerFields(self):

		for rowF in self.rowsForget:
			rowF.pack_forget()
			self.rowsForget = []

		self.neuronsEntry = []
		self.activationFunctionsEntry = []

		nextRow = True
		for fieldGroupID in range(self.numLayers):
			if nextRow:
				row = Frame(self.frameLayers)
				self.rowsForget.append(row)
				nextRow = False

			rowInner = LabelFrame(row, text = "Layer "+str(fieldGroupID+1))

			row1 = Frame(rowInner)
			widthNeurons = 15 if platform.system() == "Darwin" else 18
			lab = Label(row1, width=widthNeurons, text="Neurons:", anchor='w')
			ent = Entry(row1, width=12) #Entry Neurons
			self.neuronsEntry.append(ent)
			ent.insert(0,"0")
			lab.pack(side = LEFT, pady=(5,0), padx=(5,0))
			ent.pack(side = RIGHT, pady=(5,0), padx=(0,5), expand = YES, fill=X)

			row2 = Frame(rowInner)
			lab2 = Label(row2, width=15, text="Act. Function:", anchor='w')
			optionActVar = StringVar(value=self.activationFunctions[0])
			ent2 = OptionMenu(row2, optionActVar, *self.activationFunctions) #Entry Act Function
			self.activationFunctionsEntry.append(optionActVar)
			ent2.config(width=10)
			lab2.pack(side = LEFT, pady=(0,5), padx=(5,0))
			ent2.pack(side = RIGHT, pady=(0,5), padx=(0,5))

			row1.pack(side = TOP)
			row2.pack(side = TOP)

			expand = YES if fieldGroupID==0 else NO
			rowInner.pack(side = LEFT, fill = X, padx = 5 , pady = 5, expand = expand)

			
			if fieldGroupID%3 == 2:
				nextRow = True
				row.pack(side = TOP, fill = X, padx = 5 , pady = 5)


		if nextRow == False:
			row.pack(side = TOP, fill = X, padx = 5 , pady = 5)

	def callbackNumLayers(self, event):
		self.numLayers = int(self.dropdown.get())
		self.updateLayerFields()
		self.top.withdraw()
		self.top.update_idletasks()
		center(self.top, windowWidth=self.top.winfo_reqwidth(), windowHeight=self.top.winfo_reqheight())
		self.top.deiconify()

	def ok(self):
		self.selectedOk = True

		for field in self.fields:
			self.configuration[field] = self.entries[field].get()
		self.configuration["Total layers"] = self.numLayers

		self.configuration["Layers"] = {}
		for index, neuron, act_fn in zip(range(len(self.neuronsEntry)), self.neuronsEntry, self.activationFunctionsEntry):
			self.configuration["Layers"]["Layer "+str(index+1)] = [neuron.get(), act_fn.get()]


		self.configuration["Active Module IDs"] = []
		for index, moduleVar in zip(range(len(self.moduleList)), self.moduleList):
			if moduleVar.get() == 1:
				self.configuration["Active Module IDs"].append(index+1)

		self.top.destroy()

	def getConfiguration(self):
		if self.selectedOk:
			return self.configuration
		else:
			return -1


# ------------------------------------------------------------- 
#  ____              _      _         ____       _           
# |  _ \  __ _ _ __ (_) ___| | __ _  |  _ \ ___ (_) __ _ ___ 
# | | | |/ _` | '_ \| |/ _ \ |/ _` | | |_) / _ \| |/ _` / __|
# | |_| | (_| | | | | |  __/ | (_| | |  _ < (_) | | (_| \__ \
# |____/ \__,_|_| |_|_|\___|_|\__,_| |_| \_\___// |\__,_|___/
#                                             |__/           
# ------------------------------------------------------------- 

