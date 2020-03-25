from numpy import exp
from scipy.optimize import curve_fit
#Takes in The t,p,f data from a given combonation of
#Catheter factors and combines them into 1 data set
#Sorted by time
#Returns 2 lists of tuples sorted by time

#Dictionaries really are'nt the right data structure for this
#Since lookup is not something I'm doing here and I need the
#Output to be sorted
def combinePoints(combonation):
    flowPoints = []
    pressurePoints = []
    for run in combonation:
        #t,f,p all have the same length
        dataLen = len(run[2])
        for i in range(0,dataLen):
            timePoint = run[2][i]
            flowPoint = run[3][i]
            pressurePoint = run[4][i]

            #Add the points to the lists
            flowPoints.append((timePoint,flowPoint))
            pressurePoints.append((timePoint,pressurePoint))
    
    #Now, sort the lists by the time
    flowPoints.sort(key=lambda x: x[0])
    pressurePoints.sort(key=lambda x: x[0])
    return [flowPoints,pressurePoints]

#Unpacks a list of tuples into a list of their first element
#and a list of their second element
def unpackPoints(pointsList):
    l1 = []
    l2 = []
    for point in pointsList:
        l1.append(point[0])
        l2.append(point[1])
    return [l1,l2]

def logisticsFunc(x,L,k,x0):
    return L/(1 + exp(-k*(x-x0)))

def genNonLinPressureModel(pressurePoints):
    #initial guesses of the values of the constants:
    L = 1000 #Curve's Maximum value
    k = .5 #Logistic growth rate (steepness of the curve)
    x0 = 3 #x(time) value of the curve's midpoint
    init = [L,k,x0]
    [tData,pData] = unpackPoints(pressurePoints)
    #Put bounds on these variables since the curve fit sucks
    
    return curve_fit(logisticsFunc,tData,pData,init)

def genNonLinFlowRateModel(flowPoints):
    #initial guesses of the values of the constants:
    L = 1000 #Curve's Maximum value
    k = .5 #Logistic growth rate (steepness of the curve)
    x0 = 3 #x(time) value of the curve's midpoint
    init = [L,k,x0]
    [tData,fData] = unpackPoints(flowPoints)
    #Put bounds on these variables since the curve fit sucks

    return curve_fit(logisticsFunc,tData,fData,init)