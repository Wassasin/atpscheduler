import sklearn.svm as svm
import sklearn.ensemble as es
import numpy as np
import math as math

class StrategyScheduler(object):
    classifiers = []
    models = []

    weights = []
    strategy_mask = []

    total_time = 300.0
    time_modifier = 1.20
    prob_threshold = 0.70
    max_strategy_count = 10

    def __init__(self):
        pass

    @staticmethod
    def read(filename, include_train=True):
        X = []
        ys = []
        XNames = []
        yNames = []

        with open(filename,'r') as IS:
            firstLine = True
            for line in IS:
                if firstLine:
                    if include_train:
                        tmp = (line.strip()).split('#')
                        yNames = tmp[2].split(',')
                    firstLine = False
                    continue
                    
                tmp = (line.strip()).split('#')

                XNames.append(tmp[0])
                X.append([float(x) for x in tmp[1].split(',')])
                
                if include_train:
                    ys.append([float(x) for x in tmp[2].split(',')])

        X = np.matrix(X)
        
        if include_train:
            ys = np.matrix(ys)
            return X, ys, XNames, yNames
        else:
            return X, XNames

    def compute_weights(self, X, ys):
        weights = np.zeros(len(ys[0].A1))
        for i in range(len(ys)):
            y = ys[i].A1
            for j in range(len(y)):
                if y[j] >= 0.0:
                    weights[j] += 1
       
        return weights / np.max(weights)

    def create_mask(self, X, ys):
        mask = []

        for i in range(len(ys)):
            y = ys[i].A1

            prediction_tuples = [(j, y[j], self.weights[j]) for j in range(len(y)) if y[j] >= 0.0]
            strategies = self.schedule(prediction_tuples)

            mask.extend([j for (j, time) in strategies])

        result = [True for i in range(ys.shape[1])]
        #for i in mask:
        #    result[i] = True

        return result # sorted & unique

    def fit_file(self, filename):
        X, ys, XNames, yNames = StrategyScheduler.read(filename)
        self.fit(X, ys, yNames);
        pass
    
    def fit(self, X, ys, yNames):
        self.classifiers = []
        self.models = []
        self.yNames = yNames

        self.weights = self.compute_weights(X, ys)
        self.strategy_mask = self.create_mask(X, ys)

        for i in range(len(self.strategy_mask)):
            if not self.strategy_mask:
                self.classifiers.append(None)
                self.models.append(None)
                continue
            
            yt = ys.T[i].A1
            mask = (yt != -1.0)

            classifier = es.ExtraTreesClassifier()
            classifier.fit(X, mask)

            self.classifiers.append(classifier)

            model = es.ExtraTreesRegressor()
            model.fit(X[mask], yt[mask])

            self.models.append(model)
        pass

    def schedule(self, prediction_tuples):
        time_left = self.total_time
        strategies = []
        
        perfect_tuples = filter(lambda x: x[2] == 1.0, prediction_tuples)
        imperfect_tuples = filter(lambda x: x[2] < 1.0, prediction_tuples)

        perfect_tuples.sort(key=lambda x: x[1]) # If chance is 1.0, sort on time
        imperfect_tuples.sort(key=lambda x: -1.0 * math.log10(1.0 / (1.0 - x[2])) / (x[1] / self.total_time)) # Compute attribution to chance of success (ask Wouter for theory)

        for index, time, weight in (perfect_tuples + imperfect_tuples): # index might be either a name or an integer
            if len(strategies) == self.max_strategy_count:
                break;

            if time < 0.5: # Faster than 0.0 sec should be a good strategy (note that -1 is filtered out above)
                time = 0.5

            time = time * self.time_modifier

            if time_left < time: # If does not fit, try next in line
               continue

            strategies.append((index, time))
            time_left -= time
        
        strategies.sort(key=lambda x: x[1])

        for i in range(len(strategies)):
            strategies[i] = (strategies[i][0], strategies[i][1] + time_left / len(strategies))

        return strategies

    def schedule_to_string(self, strategies):
        return ",".join(['%s:%f' % strategy for strategy in strategies])

    def predict(self, features):
        prediction_tuples = []
        for i in np.where(self.strategy_mask)[0]:
            falseProb, trueProb = self.classifiers[i].predict_proba(features)[0] # Classes are ordered by arithmetical order
            if trueProb >= self.prob_threshold:
                prediction_tuples.append((self.yNames[i], self.models[i].predict(features), self.weights[i] * trueProb))

        if len(prediction_tuples) == 0: # Got no viable solution
            for i in np.where(self.strategy_mask)[0]:
                prediction_tuples.append((self.yNames[i], self.models[i].predict(features), self.weights[i]))
        
        strategies = self.schedule(prediction_tuples)
        
        if len(strategies) == 0:
            strategies = [('NewStrategy101164', 150.0), ('NewStrategy101980', 150.0)] # Just try something
        
        return self.schedule_to_string(strategies)
    
    def analyze(self, X, ys):
        N,M = X.shape
        
        for i in range(N):
            print "%i:" % i
            features = np.array(X[i].A1)
            y = np.array(ys[i].A1)
            
            prediction_tuples = []
            for j in np.where(self.strategy_mask)[0]:
                falseProb, trueProb = self.classifiers[j].predict_proba(features)[0] # Classes are ordered by arithmetical order
                if trueProb >= self.prob_threshold:
                    prediction_tuples.append((j, self.models[j].predict(features), self.weights[j] * trueProb))

            if len(prediction_tuples) == 0: # Got no viable solution
                for j in np.where(self.strategy_mask)[0]:
                    prediction_tuples.append((j, self.models[j].predict(features), self.weights[j]))
            
            strategies = self.schedule(prediction_tuples)
            
            current_time = 0.0
            success = False
            for j, time in strategies:
                current_time += time
            
                if y[j] == -1:
                    continue
                
                if time < y[j]:
                    print "Aborted too soon! (diff %f, length %f)" % (y[j] - time, y[j])
                    continue
                else:
                    print "Success (length %f, diff %f, total %f)" % (time, time - y[j], current_time - time)
                    success = True
                    break
                
            if not success:
                print "Failure (options: total %i, attempted %i)" % (len(np.argwhere(y != -1)), len(strategies))
        pass
