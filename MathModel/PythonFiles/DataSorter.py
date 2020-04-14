def insertCombonation(combonations,entry):
    for combo in combonations:
        gauge = combo[0][0]
        otherInfo = combo[0][1]
        if(entry[0] == gauge and entry[1] == otherInfo):
            combo.append(entry)
            break
        if(entry[0] == None or entry[1] == None):
            break

def getEntry(parsedData,i):
    r = []
    for dataType in parsedData:
        r.append(dataType[i])
    return r

def dataSorter(parsedData):
    seen = []
    combonations = []
    #The length of each list in parsedData
    #Is the total number of data points recorded
    for i in range(len(parsedData[0])):
        gauge = parsedData[0][i]
        otherInfo = parsedData[1][i]
        if(([gauge,otherInfo] in seen)):
            entry = getEntry(parsedData,i)
            insertCombonation(combonations,entry)
        else:
            seen.append([gauge,otherInfo])
            entry = getEntry(parsedData,i)
            combonations.append([entry])
    
    #Combonations is a List of Lists,
    #Each list has the same gauge, otherInfo, but different t,f,p
    #Feed combonations to nonlinear data analysis
    return combonations