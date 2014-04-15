import logging
import sklearn.linear_model as lm
import sklearn.ensemble as es
import numpy as np

from MLiP_eval import StrategyScheduleScore

class StrategyScheduler(object):
    classifiers = []
    models = []
    
    attributeNames = []
    yNames = []
    
    total_time = 300.0
    time_modifier = 1.20
    max_strategy_count = 5

    def __init__(self):
        pass
    
    def read(self, filename):
        names = []
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

                names.append(tmp[0])
                X.append([float(x) for x in tmp[1].split(',')])
                ys.append([float(x) for x in tmp[2].split(',')])

        X = np.matrix(X)
        ys = np.matrix(ys)
        
        return X, ys, names
    
    def create_mask(self, X, ys):
        mask = []
        for i in range(len(ys)):
            y = ys[i].A1

            prediction_tuples = [(j, y[j]*self.time_modifier) for j in range(len(y)) if y[j] >= 0.0]
            strategies = self.schedule(prediction_tuples)
            
            mask.extend([i for (i, time) in strategies])
        
        return np.unique(mask) # sorted & unique
    
    def fit(self, filename):
        self.classifiers = []
        self.models = []
        
        X, ys, names = self.read(filename)
        strategy_mask = self.create_mask(X, ys)
        
        # make dataset consistent
        self.yNames = [self.yNames[i] for i in strategy_mask]
        ys = np.matrix([ys.T[i].A1 for i in strategy_mask]).T
        
        i = 0
        for y in ys.T:
            print i
            i += 1
            y = y.A1
            mask = (y != -1.0)

            classifier = es.RandomForestClassifier()
            classifier.fit(X, mask)

            self.classifiers.append(classifier)

            model = lm.LinearRegression()
            model.fit(X[mask], y[mask])

            self.models.append(model)
           
        pass
    
    def schedule(self, prediction_tuples):
        time_left = self.total_time
        strategies = []
        
        prediction_tuples.sort(key=lambda x: x[1])
        for index, time in prediction_tuples: # index might be either a name or an integer
            if len(strategies) == self.max_strategy_count:
                break;
            
            if time < 0.0: # Faster than 0.0 sec should be a good strategy (note that -1 is filtered out above)
                time = 0.1
        
            time = time * self.time_modifier
            
            if time_left < time:
                if time_left == self.total_time:
                    strategies.append((index, time_left)) # Try anyway
                    time_left = 0.0
                break
            
            strategies.append((index, time))
            time_left -= time
        
        for i in range(len(strategies)):
            strategies[i] = (strategies[i][0], strategies[i][1] + time_left / len(strategies))
        
        return strategies
        
    def schedule_to_string(self, strategies):
        return ",".join(['%s:%f' % strategy for strategy in strategies])
        
    def predict(self, features):
        prediction_tuples = []
        for i in range(len(self.models)):
            if self.classifiers[i].predict(features):
                prediction_tuples.append((self.yNames[i], self.models[i].predict(features)))
        
        if len(prediction_tuples) > 0: 
            strategies = self.schedule(prediction_tuples)
        else: # Got no viable solution
            strategies = [('NewStrategy101164', 150.0), ('NewStrategy101980', 150.0)] # Just try something
    
        return self.schedule_to_string(strategies)

if __name__ == '__main__':
    trainFile = 'orig/MLiP_train' 
    testFile = 'orig/MLiP_train'
    mySchedule = 'My_MLiP_train_example_schedule'
    
    SS = StrategyScheduler()
    
    SS.fit(trainFile)
    
    # Get the test problems
    testData = []
    with open(testFile,'r') as IS:
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
    Eval = StrategyScheduleScore(testFile)
    solved, score = Eval.eval_schedules(mySchedule)
    print 'Solved: %s %%' % solved                    
    print 'Score: %s %%  (%s / %s)' % (round(100*score/Eval.bestScore,4),score,Eval.bestScore)
