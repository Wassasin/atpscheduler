'''
Template for a strategy schedule predictor.

Created on Mar 23, 2014

@author: Daniel Kuehlwein
'''

import logging
from MLiP_eval import StrategyScheduleScore

class StrategyScheduler(object):
    '''
    Template for a strategy scheduler.
    '''

    def __init__(self):
        '''
        Initialize whatever needs to be initialized
        '''
        pass

    def fit(self,trainData):
        '''
        Fit your model to the training data
        '''
        pass
    def predict(self,features):
        '''
        Use the features to predict the best strategy schedule possible
        '''
        return 'NewStrategy101164:150.0,NewStrategy101980:150.0'

"""
Example use
"""    
if __name__ == '__main__':
    trainData = 'MLiP_train' 
    mySchedule = 'My_MLiP_train_example_schedule'
    
    SS = StrategyScheduler()
    SS.fit(trainData)
    
    # Get the test problems
    testData = []
    with open(trainData,'r') as IS:
        firstLine = True
        for line in IS:
            if firstLine:
                firstLine = False
                continue
            tmp = (line.strip()).split('#')
            pName = tmp[0]
            pFeatures = [float(x) for x in tmp[1].split(',')]
            testData.append((pName,pFeatures))
    
    # Create schedules
    with open(mySchedule,'w') as OS:
        for pName,pFeatures in testData:
            schedule = SS.predict(pFeatures)
            OS.write('%s#%s\n' % (pName,schedule))
            
    # Evaluate schedule 
    Eval = StrategyScheduleScore(trainData)
    solved, score = Eval.eval_schedules(mySchedule)
    print 'Solved: %s %%' % solved                    
    print 'Score: %s %%  (%s / %s)' % (round(100*score/Eval.bestScore,4),score,Eval.bestScore)
    