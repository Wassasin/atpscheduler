from StrategyScheduler import StrategyScheduler
from MLiP_eval import StrategyScheduleScore

if __name__ == '__main__':
    trainFile = 'orig/MLiP_train'
    
    X, ys, XNames, yNames = StrategyScheduler.read(trainFile)
    
    SS = StrategyScheduler()
    SS.fit(X, ys, yNames)
    
    SS.analyze(X, ys)
