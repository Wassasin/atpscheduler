import sklearn.linear_model as lm
import sklearn.ensemble as es
import numpy as np
import math as math

class StrategyScheduler(object):
    classifiers = []
    models = []
    yNames = []
    
    total_time = 300.0
    time_modifier = 1.10
    max_strategy_count = 20

    def __init__(self):
        pass
    
    @staticmethod
    def read(filename):
        X = []
        ys = []
        XNames = []
        yNames = []
    
        with open(filename,'r') as IS:
            firstLine = True
            for line in IS:
                if firstLine:
                    tmp = (line.strip()).split('#')
                    yNames = tmp[2].split(',')
                
                    firstLine = False
                    continue
                tmp = (line.strip()).split('#')

                XNames.append(tmp[0])
                X.append([float(x) for x in tmp[1].split(',')])
                ys.append([float(x) for x in tmp[2].split(',')])

        X = np.matrix(X)
        ys = np.matrix(ys)
        
        return X, ys, XNames, yNames
    
    def create_mask(self, X, ys):
        mask = []
        for i in range(len(ys)):
            y = ys[i].A1

            prediction_tuples = [(j, y[j]*self.time_modifier) for j in range(len(y)) if y[j] >= 0.0]
            strategies = self.schedule(prediction_tuples)
            
            mask.extend([i for (i, time) in strategies])
        
        return np.unique(mask) # sorted & unique
    
    def fit_file(self, filename):
        X, ys, XNames, yNames = StrategyScheduler.read(filename)
        self.fit(X, ys, yNames);
        pass
    
    def fit(self, X, ys, yNames):
        self.classifiers = []
        self.models = []
        self.yNames = yNames
        
        strategy_mask = self.create_mask(X, ys)
        
        # make dataset consistent
        self.yNames = [self.yNames[i] for i in strategy_mask]
        ys = np.matrix([ys.T[i].A1 for i in strategy_mask]).T
        
        for yt in ys.T:
            yt = yt.A1
            mask = (yt != -1.0)

            classifier = es.RandomForestClassifier()
            classifier.fit(X, mask)

            self.classifiers.append(classifier)

            model = lm.LinearRegression()
            model.fit(X[mask], yt[mask])

            self.models.append(model)
           
        pass
    
    def schedule(self, prediction_tuples):
        time_left = self.total_time
        strategies = []
        
        prediction_tuples.sort(key=lambda x: x[1])
        for index, time in prediction_tuples: # index might be either a name or an integer
            if len(strategies) == self.max_strategy_count:
                break;
            
            if time < 0.5: # Faster than 0.0 sec should be a good strategy (note that -1 is filtered out above)
                time = 0.5
        
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
