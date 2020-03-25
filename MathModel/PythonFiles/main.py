from DataParser import openFile
from DataSorter import dataSorter
import NonLinAnalyzer2 as NLA
#Note, since you are working in a virtual environment for python, you have to install each of
#The required packages into the venv itself
import matplotlib.pyplot as plt
import numpy as np

def main():
    data = openFile("RawData.txt")
    #print("data = %s" % (data))
    combonations = dataSorter(data)

    num = 2 #Some combonation
    #T,f,p data for the xth combonation
    [flowPoints,pressurePoints] = NLA.combinePoints(combonations[num])
    #Generate the Model for combonations[x]
    popt, y = NLA.genNonLinPressureModel(pressurePoints)
    [L,k,x0] = popt

    #Now, compare it to some real data:
    tx = combonations[num][0][2]
    px = combonations[num][0][4]
    print("tx = %s" % (tx))
    print("px = %s" % (px))

    x = np.linspace(0,30,100)
    y = L/(1 + np.exp(-k*(x-x0)))

    plt.plot(x, y,'r--',linewidth=2.0)
    plt.plot(tR,px,'b--',linewidth=2.0)
    plt.xlabel("time (min)")
    plt.ylabel("pressure of in catheter (kPa)")
    plt.title("Fit Logistics Curve for combonations[%s]"%(num))
    plt.show()


main()