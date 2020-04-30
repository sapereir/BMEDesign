# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bmeGUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(814, 555)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.simulateNN = QtWidgets.QPushButton(self.centralwidget)
        self.simulateNN.setGeometry(QtCore.QRect(570, 60, 171, 41))
        self.simulateNN.setAutoFillBackground(False)
        self.simulateNN.setStyleSheet("background-color: rgb(184, 161, 255);")
        self.simulateNN.setObjectName("simulateNN")
        self.simulateMathModel = QtWidgets.QPushButton(self.centralwidget)
        self.simulateMathModel.setGeometry(QtCore.QRect(380, 60, 171, 41))
        self.simulateMathModel.setAutoFillBackground(False)
        self.simulateMathModel.setStyleSheet("background-color: rgb(183, 224, 255);")
        self.simulateMathModel.setObjectName("simulateMathModel")
        self.lineGraphView = QtWidgets.QGraphicsView(self.centralwidget)
        self.lineGraphView.setGeometry(QtCore.QRect(410, 160, 351, 221))
        self.lineGraphView.setObjectName("lineGraphView")
        self.inputFile = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.inputFile.setGeometry(QtCore.QRect(40, 70, 271, 31))
        self.inputFile.setObjectName("inputFile")
        self.results_log = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.results_log.setGeometry(QtCore.QRect(40, 180, 331, 201))
        self.results_log.setStyleSheet("background-color: rgb(252, 227, 255);")
        self.results_log.setObjectName("results_log")
        self.console_log = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.console_log.setGeometry(QtCore.QRect(40, 430, 721, 101))
        self.console_log.setStyleSheet("background-color: rgb(214, 254, 255);")
        self.console_log.setObjectName("console_log")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 410, 141, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 160, 141, 21))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.simulateNN.setText(_translate("MainWindow", "Neural Network"))
        self.simulateMathModel.setText(_translate("MainWindow", "Math Model"))
        self.inputFile.setPlainText(_translate("MainWindow", "Path to file\n"
""))
        self.label.setText(_translate("MainWindow", "Console Log:"))
        self.label_2.setText(_translate("MainWindow", "Results:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
