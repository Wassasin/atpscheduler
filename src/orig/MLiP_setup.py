'''
Created on Mar 23, 2014

@author: Daniel Kuehlwein
'''

import logging
import os
from readData import get_e_features
from Strategy import load_strategies
from random import shuffle

PATH = '/scratch/kuehlwein/males/E'

def create_data_file(fileName,problemList):
    with open(fileName,'w') as OS:
        OS.write(' #')
        OS.write(','.join(['f'+str(i) for i in range(fLength)]))
        OS.write('#')
        OS.write(','.join(names))
        OS.write('\n')
        for p in problemList:    
            OS.write(p+'#')
            OS.write(','.join([str(f) for f in featureDict[p]]))
            OS.write('#')
            p_extended = '/scratch/kuehlwein/TPTP-v5.4.0/' + p
            stratTimes = []
            for n in names:
                s = stratDict[n]
                try:
                    stratTimes.append(str(s.solvedProblems[p_extended]))
                except:
                    stratTimes.append('-1')
            OS.write(','.join(stratTimes))
            OS.write('\n')

logging.basicConfig(level=logging.INFO,
                    format='%% %(message)s',
                    datefmt='%d-%m %H:%M:%S')
logger = logging.getLogger('MLiP Setup')

# Create Train/Test problems
problemFile = os.path.join(PATH,'data','CASC24Training')
problems = []
with open(problemFile,'r') as pFile:
    for p in pFile:
        problems.append((p.strip()))
shuffle(problems)
problemsTrain = problems[:900]
problemsTest = problems[900:]

stratFolder = os.path.join(PATH,'results')
strategies = load_strategies(stratFolder)
names = sorted([s.name for s in strategies])
stratDict = {}
for s in strategies:
    stratDict[s.name] = s

featureDict = {}
#problemsTrain = problemsTrain[:30] 
for p in problems:
    featureDict[p] = get_e_features(p)
fLength = len(featureDict[problems[0]])

create_data_file('MLiP_train',problemsTrain)
create_data_file('MLiP_test',problemsTest)

with open('MLiP_train_example_schedule','w') as OS:
    for p in problemsTrain:
        OS.write(p+'#NewStrategy101164:150.0,NewStrategy101980:150.0')
        OS.write('\n')

with open('MLiP_test_features','w') as OS:
    for p in problemsTest:
        OS.write(p+'#')
        OS.write(','.join([str(f) for f in featureDict[p]]))
        OS.write('\n')
