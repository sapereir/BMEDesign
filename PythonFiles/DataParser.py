import re

#Returns an integer with the gauge number taken from the gauge string in the
#Large text fiel
def parseGaugeString(gauge):
    if(gauge == "Unknown"):
        #Lots of the entries are marked as 'unknown',
        #So I will keep track of these with 0 entries
        return 0
    numString = gauge[-2:]
    if(numString.isdigit()):
        return int(numString)
    else:
        #If the catheter has something out of the ordinary
        #For the number
        return 0

def parseOtherInfo(otherInfo):
    segments = otherInfo.strip().split(' ')
    information = []
    for info in segments:
        modifiedInfo = info.replace('@','').replace('C','').replace('S','')
        information.append(modifiedInfo)
    return information

#Opens a Bayer pressure injector file and returns the [time,flowRate,pressure] array
def openFile(fileName):
    with open (fileName,"rt") as data: #"rt" is for reading text
        content = data.readlines()

        #The collections of all the times for all the injections are labeled 'runs'
        timeRuns = []
        flowRateRuns = []
        pressureRuns = []
        gaugeRuns = []
        otherInfoRuns = []
        for lineNum in range(len(content)):
            print("lineNum = %s" % (lineNum))
            timeList = []
            flowRateList = []
            pressureList = []

            entries = re.split('\t', content[lineNum])
            gauge = entries[3]
            otherInfo = entries[5]
            injectionData = entries[11].replace(" ","").replace('[','').replace(']','').strip()
            injectionDataArray = injectionData.split(';')

            timeList = injectionDataArray[0].split(',')
            flowRateList = injectionDataArray[1].split(',')
            pressureList = injectionDataArray[2].split(',')

            for entry in timeList:
                entry = float(entry)

            for entry in flowRateList:
                entry = float(entry)

            for entry in pressureList:
                entry = float(entry)
            timeRuns.append(timeList)
            flowRateRuns.append(flowRateList)
            pressureRuns.append(pressureList)
            gaugeRuns.append(parseGaugeString(gauge))
            otherInfoRuns.append(parseOtherInfo(otherInfo))

    #In the future, return more different attributes for each injection
    #In a list maintained alongside the output below.
    return [gaugeRuns,otherInfoRuns,timeRuns,flowRateRuns,pressureRuns]