from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from datetime import datetime
import bmeGUI
import pyqtgraph as pg
import callMatlabInPython

class ExampleApp(QtWidgets.QMainWindow, bmeGUI.Ui_MainWindow):
	def __init__(self, parent=None):
		super(ExampleApp, self).__init__(parent)
		self.setupUi(self)
		self.simulateNN.clicked.connect(self.simulate_neuralNet)
		self.simulateMathModel.clicked.connect(self.simulate_mathModel)
		self.runMatLabModel.clicked.connect(self.start_MatLabModel)
		self.displayPressureGraph.clicked.connect(self.update_graph_pressure)
		self.displayFlowGraph.clicked.connect(self.update_graph_flow)
		self.displayQerr.clicked.connect(self.update_graph_q_err)
		self.displayPressureGraph.setEnabled(False)
		self.displayFlowGraph.setEnabled(False)
		self.displayQerr.setEnabled(False)
		self.time, self.pressure, self.flow, self.q_err = [], [], [], []

		self.data = []

	def simulate_neuralNet(self):
		print('in simulate_neuralnet')
		self.stackedWidget.setCurrentIndex(0)
		current_time = datetime.now().strftime("%H:%M:%S")
		self.console_log.setPlainText(current_time+": "+ "Switching to Neural Network Simulation\n" + self.console_log.toPlainText())

	def simulate_mathModel(self):
		print('in simulate_mathModel')
		self.stackedWidget.setCurrentIndex(1)
		current_time = datetime.now().strftime("%H:%M:%S")
		self.console_log.setPlainText(current_time+": "+ "Switching to MatLab Simulation\n" + self.console_log.toPlainText())

	def start_MatLabModel(self):
		self.setupGraph()
		current_time = datetime.now().strftime("%H:%M:%S")
		self.console_log.setPlainText(current_time+": "+ "Running MatLab Model\n" + self.console_log.toPlainText())
		# function that retrieves mat lab data from callMatlabInPython is here
		self.time, self.pressure, self.flow, self.q_err = callMatlabInPython.retrieveMatLab()
		# wait for the matlab function to finish calculating the values
		while self.time is None or self.pressure is None or self.flow is None or self.q_err is None:
			pass
		self.displayPressureGraph.setEnabled(True)
		self.displayFlowGraph.setEnabled(True)
		self.displayQerr.setEnabled(True)


	def setupGraph(self):
		self.lineGraphView.setBackground('w')
		self.pen = pg.mkPen(color=(0,0,255))
		self.lineGraphView.setLabel('left', 'Time')

	def update_graph_pressure(self):
		self.lineGraphView.clear()
		self.lineGraphView.setLabel('bottom', 'Pressure')
		self.lineGraphView.plot(self.time, self.pressure, pen=self.pen)

	def update_graph_flow(self):
		self.lineGraphView.clear()
		self.lineGraphView.setLabel('bottom', 'Flow')
		self.lineGraphView.plot(self.time,self.flow, pen=self.pen)

	def update_graph_q_err(self):
		self.lineGraphView.clear()
		self.lineGraphView.setLabel('bottom', 'Q_err')
		self.lineGraphView.plot(self.time,self.q_err, pen=self.pen)


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