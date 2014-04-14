import logging
import sklearn.linear_model as lm
import numpy as np

from MLiP_eval import StrategyScheduleScore

class StrategyScheduler(object):
    models = []
    attributeNames = []
    yNames = []
    
    failure_time = 600.0
    total_time = 300.0
    time_modifier = 1.20

    def __init__(self):
        pass
    def fit(self, filename):
        X = []
        ys = []
    
        with open(filename,'r') as IS:
            firstLine = True
            for line in IS:
                if firstLine:
                    tmp = (line.strip()).split('#')
                    self.attributeNames = tmp[1].split(',')
                    self.yNames = tmp[2].split(',')
                
                    firstLine = False
                    continue
                tmp = (line.strip()).split('#')

                X.append([float(x) for x in tmp[1].split(',')])
                ys.append([float(x) if x != "-1" else self.failure_time for x in tmp[2].split(',')])

        X = np.matrix(X)
        ys = np.matrix(ys)
        
        for y in ys.T:
           y = y.A1
           
           model = lm.LinearRegression()
           model.fit(X, y)
           
           self.models.append(model)
           
        pass
    def predict(self,features):
        prediction_tuples = []
        for i in range(len(self.models)):
            prediction_tuples.append((self.yNames[i], self.models[i].predict(features)))
        prediction_tuples.sort(key=lambda x: x[1])
        
        time_left = self.total_time
        strategies = []
        for name, time in prediction_tuples:
            if time < 0.0: # Faster than 0.0 sec should be a good strategy (note that -1 is filtered out above)
                time = 1.0;
        
            time = time * self.time_modifier
            
            if time_left < time:
                if time_left == self.total_time:
                    strategies.append((name, self.total_time)) # Try anyway
                break
            
            strategies.append((name, time))
            time_left -= time
        
        strategies[0] = (strategies[0][0], strategies[0][1] + time_left) # Give time left to best strategy
    
        return ",".join(['%s:%f' % strategy for strategy in strategies])

if __name__ == '__main__':
    trainData = 'orig/MLiP_train' 
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
