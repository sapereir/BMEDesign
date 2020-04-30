from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from datetime import datetime
import bmeGUI
import pyqtgraph as pg

class ExampleApp(QtWidgets.QMainWindow, bmeGUI.Ui_MainWindow):
	def __init__(self, parent=None):
		super(ExampleApp, self).__init__(parent)
		self.setupUi(self)
		self.simulateNN.clicked.connect(self.simulate_neuralNet)
		self.simulateMathModel.clicked.connect(self.simulate_mathModel)
		

	def simulate_neuralNet(self):
		print('in simulate_neuralnet')
		current_time = datetime.now().strftime("%H:%M:%S")
		self.console_log.setPlainText(current_time+": "+ "Start Neural Network simulation\n" + self.console_log.toPlainText())


	def simulate_mathModel(self):
		print('in simulate_mathModel')
		current_time = datetime.now().strftime("%H:%M:%S")
		self.console_log.setPlainText(current_time+": "+ "Start Math Model simulation \n" + self.console_log.toPlainText())


def main():
    app = QApplication(sys.argv)
    #appctxt = ApplicationContext()
    form = ExampleApp()
    form.show()
    form.raise_()
    app.exec_()
    #exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    #sys.exit(exit_code)


if __name__ == '__main__':
    main()