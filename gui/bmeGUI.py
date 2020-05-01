# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bmeGUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(814, 555)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.simulateNN = QtWidgets.QPushButton(self.centralwidget)
        self.simulateNN.setGeometry(QtCore.QRect(380, 30, 171, 41))
        self.simulateNN.setAutoFillBackground(False)
        self.simulateNN.setStyleSheet("background-color: rgb(184, 161, 255);")
        self.simulateNN.setObjectName("simulateNN")
        self.simulateMathModel = QtWidgets.QPushButton(self.centralwidget)
        self.simulateMathModel.setGeometry(QtCore.QRect(170, 30, 171, 41))
        self.simulateMathModel.setAutoFillBackground(False)
        self.simulateMathModel.setStyleSheet("background-color: rgb(183, 224, 255);")
        self.simulateMathModel.setObjectName("simulateMathModel")
        self.console_log = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.console_log.setEnabled(False)
        self.console_log.setGeometry(QtCore.QRect(40, 430, 721, 101))
        self.console_log.setStyleSheet("background-color: rgb(214, 254, 255);")
        self.console_log.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.console_log.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.console_log.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.console_log.setObjectName("console_log")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 410, 141, 21))
        self.label.setObjectName("label")
        self.stackedWidget_2 = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget_2.setGeometry(QtCore.QRect(20, 120, 731, 261))
        self.stackedWidget_2.setObjectName("stackedWidget_2")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.stackedWidget_2.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.stackedWidget_2.addWidget(self.page_4)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(30, 90, 741, 301))
        self.stackedWidget.setStyleSheet("")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.constrastInput = QtWidgets.QPlainTextEdit(self.page)
        self.constrastInput.setGeometry(QtCore.QRect(140, 10, 201, 21))
        self.constrastInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.constrastInput.setObjectName("constrastInput")
        self.label_3 = QtWidgets.QLabel(self.page)
        self.label_3.setGeometry(QtCore.QRect(60, 10, 60, 16))
        self.label_3.setObjectName("label_3")
        self.salineInput = QtWidgets.QPlainTextEdit(self.page)
        self.salineInput.setGeometry(QtCore.QRect(140, 40, 201, 21))
        self.salineInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.salineInput.setObjectName("salineInput")
        self.label_4 = QtWidgets.QLabel(self.page)
        self.label_4.setGeometry(QtCore.QRect(60, 40, 60, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.page)
        self.label_5.setGeometry(QtCore.QRect(60, 70, 60, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.page)
        self.label_6.setGeometry(QtCore.QRect(60, 100, 60, 16))
        self.label_6.setObjectName("label_6")
        self.mixedInput = QtWidgets.QPlainTextEdit(self.page)
        self.mixedInput.setGeometry(QtCore.QRect(140, 70, 201, 21))
        self.mixedInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.mixedInput.setObjectName("mixedInput")
        self.amountCInput = QtWidgets.QPlainTextEdit(self.page)
        self.amountCInput.setGeometry(QtCore.QRect(140, 100, 201, 21))
        self.amountCInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.amountCInput.setObjectName("amountCInput")
        self.amountSInput = QtWidgets.QPlainTextEdit(self.page)
        self.amountSInput.setGeometry(QtCore.QRect(140, 130, 201, 21))
        self.amountSInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.amountSInput.setObjectName("amountSInput")
        self.label_7 = QtWidgets.QLabel(self.page)
        self.label_7.setGeometry(QtCore.QRect(60, 160, 60, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.page)
        self.label_8.setGeometry(QtCore.QRect(60, 130, 60, 16))
        self.label_8.setObjectName("label_8")
        self.amountMInput = QtWidgets.QPlainTextEdit(self.page)
        self.amountMInput.setGeometry(QtCore.QRect(140, 160, 201, 21))
        self.amountMInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.amountMInput.setObjectName("amountMInput")
        self.percentMInput = QtWidgets.QPlainTextEdit(self.page)
        self.percentMInput.setGeometry(QtCore.QRect(140, 190, 201, 21))
        self.percentMInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.percentMInput.setObjectName("percentMInput")
        self.label_9 = QtWidgets.QLabel(self.page)
        self.label_9.setGeometry(QtCore.QRect(60, 190, 60, 16))
        self.label_9.setObjectName("label_9")
        self.neuralNetResults = QtWidgets.QPlainTextEdit(self.page)
        self.neuralNetResults.setGeometry(QtCore.QRect(410, 70, 251, 81))
        self.neuralNetResults.setStyleSheet("background-color: rgb(206, 224, 255);")
        self.neuralNetResults.setObjectName("neuralNetResults")
        self.label_10 = QtWidgets.QLabel(self.page)
        self.label_10.setGeometry(QtCore.QRect(50, 220, 60, 16))
        self.label_10.setObjectName("label_10")
        self.flowRateInput = QtWidgets.QPlainTextEdit(self.page)
        self.flowRateInput.setGeometry(QtCore.QRect(140, 220, 201, 21))
        self.flowRateInput.setStyleSheet("background-color: rgb(229, 226, 255);")
        self.flowRateInput.setObjectName("flowRateInput")
        self.predictPressure = QtWidgets.QPushButton(self.page)
        self.predictPressure.setGeometry(QtCore.QRect(490, 170, 151, 31))
        self.predictPressure.setAutoFillBackground(False)
        self.predictPressure.setStyleSheet("background-color: rgb(255, 230, 136);")
        self.predictPressure.setObjectName("predictPressure")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.lineGraphView = PlotWidget(self.page_2)
        self.lineGraphView.setGeometry(QtCore.QRect(380, 40, 351, 221))
        self.lineGraphView.setObjectName("lineGraphView")
        self.label_2 = QtWidgets.QLabel(self.page_2)
        self.label_2.setGeometry(QtCore.QRect(30, 20, 141, 21))
        self.label_2.setObjectName("label_2")
        self.runMatLabModel = QtWidgets.QPushButton(self.page_2)
        self.runMatLabModel.setGeometry(QtCore.QRect(100, 50, 161, 31))
        self.runMatLabModel.setAutoFillBackground(False)
        self.runMatLabModel.setStyleSheet("background-color: rgb(183, 224, 255);")
        self.runMatLabModel.setObjectName("runMatLabModel")
        self.displayPressureGraph = QtWidgets.QPushButton(self.page_2)
        self.displayPressureGraph.setGeometry(QtCore.QRect(90, 100, 181, 31))
        self.displayPressureGraph.setAutoFillBackground(False)
        self.displayPressureGraph.setStyleSheet("background-color: rgb(183, 224, 255);")
        self.displayPressureGraph.setObjectName("displayPressureGraph")
        self.displayFlowGraph = QtWidgets.QPushButton(self.page_2)
        self.displayFlowGraph.setGeometry(QtCore.QRect(90, 150, 181, 31))
        self.displayFlowGraph.setAutoFillBackground(False)
        self.displayFlowGraph.setStyleSheet("background-color: rgb(183, 224, 255);")
        self.displayFlowGraph.setObjectName("displayFlowGraph")
        self.displayQerr = QtWidgets.QPushButton(self.page_2)
        self.displayQerr.setGeometry(QtCore.QRect(90, 200, 181, 31))
        self.displayQerr.setAutoFillBackground(False)
        self.displayQerr.setStyleSheet("background-color: rgb(183, 224, 255);")
        self.displayQerr.setObjectName("displayQerr")
        self.stackedWidget.addWidget(self.page_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.simulateNN.setText(_translate("MainWindow", "Neural Network"))
        self.simulateMathModel.setText(_translate("MainWindow", "Math Model"))
        self.label.setText(_translate("MainWindow", "Console Log:"))
        self.label_3.setText(_translate("MainWindow", "Contrast:"))
        self.label_4.setText(_translate("MainWindow", "Saline:"))
        self.label_5.setText(_translate("MainWindow", "Mixed:"))
        self.label_6.setText(_translate("MainWindow", "AmountC:"))
        self.label_7.setText(_translate("MainWindow", "AmountM: "))
        self.label_8.setText(_translate("MainWindow", "AmountS: "))
        self.label_9.setText(_translate("MainWindow", "% M: "))
        self.neuralNetResults.setPlainText(_translate("MainWindow", "Predicted Pressure: "))
        self.label_10.setText(_translate("MainWindow", "Flow Rate:"))
        self.predictPressure.setText(_translate("MainWindow", "Predict Pressure"))
        self.label_2.setText(_translate("MainWindow", "Results:"))
        self.runMatLabModel.setText(_translate("MainWindow", "Run MatLab Model"))
        self.displayPressureGraph.setText(_translate("MainWindow", "Display Prssure Graph"))
        self.displayFlowGraph.setText(_translate("MainWindow", "Display Flow Graph"))
        self.displayQerr.setText(_translate("MainWindow", "Display q_err"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
