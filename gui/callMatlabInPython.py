#Note that this module requires python 3.7, 3.6 or 2.7, 
#CANNOT run on Python 3.8, thus compile with python 3.6
#Python 3.7 give a very strange error with DLL files, got it working with 3.6!

#Also, requires that you have a valid, liscences copy of MATLAB
#On your computer

#To use matlab to install the API, follow the instructions at:
#https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

import matlab.engine

#Note, the way I got this to work is to have both the python and matlab functions
#That you will be using all in the same folder -> this will make it easier to run
#Whatever your model outputs in Python

def retrieveMatLab():
	# time = list(range(0, 10))
	# pressure = list(range(0, 100,10))
	# flow = list(range(5, 55, 5))
	# q_err = list(range(0,50, 5))
	eng = matlab.engine.start_matlab()
	time,pressure,flow,q_err = eng.StepModelForPython(nargout=4)
	return time, pressure, flow, q_err