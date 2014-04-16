from StrategyScheduler import StrategyScheduler
from MLiP_eval import StrategyScheduleScore
from multiprocessing import Pool
import sklearn.cross_validation as skcv

def evalFold(arg):
    train_index, test_index = arg
    
    SS = StrategyScheduler()
    SS.fit(X[train_index], ys[train_index], yNames)
    
    success_count = 0
    for i in test_index:
        success, score = SSS.eval_schedule('%s#%s' % (XNames[i], SS.predict(X[i])))
        if success:
            success_count += 1
            
    print 'Current: %i / %i' % (success_count, len(test_index))
    return (success_count, len(test_index))

if __name__ == '__main__':
    trainFile = 'orig/MLiP_train' 
    
    SSS = StrategyScheduleScore(trainFile)
    X, ys, XNames, yNames = StrategyScheduler.read(trainFile)
    N,M = X.shape
    
    kf = skcv.KFold(n=N, n_folds=N-1)
    
    success_count = 0
    total_count = 0
    
    pool = Pool()
    for success_sub, total_sub in pool.map(evalFold, kf):
        success_count += success_sub
        total_count += total_sub
    
    print 'Solved: %s %%' % round(100.0 * success_count / total_count, 2)
