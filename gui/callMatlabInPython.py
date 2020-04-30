#Note that this module requires python 3.7 or 2.7, 
#CANNOT run on Python 3.8, thus compile with python 3.7 

#Also, requires that you have a valid, liscences copy of MATLAB
#On your computer

#To use matlab to install the API, follow the instructions at:
#https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
import matlab.engine
eng = matlab.engine.start_matlab()

[time,pressure,flow,q_err] = eng.StepModel(nargout=4)
print(time)

#Note, the way I got this to work is to have both the python and matlab functions
#That you will be using all in the same folder -> this will make it easier to run
#Whatever your model outputs in Python