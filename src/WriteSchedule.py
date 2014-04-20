from StrategyScheduler import StrategyScheduler
from MLiP_eval import StrategyScheduleScore

if __name__ == '__main__':
    trainFile = 'orig/MLiP_train' 
    testFile = 'orig/MLiP_train'
    mySchedule = 'My_MLiP_train_example_schedule'
    
    SS = StrategyScheduler()
    SS.fit_file(trainFile)
    
    X, XNames = StrategyScheduler.read(testFile, False)
    N,M = X.shape
    
    # Create schedules
    with open(mySchedule,'w') as OS:
        for i in range(N):
            schedule = SS.predict(X[i])
            OS.write('%s#%s\n' % (XNames[i], schedule))

    # Evaluate schedule 
    Eval = StrategyScheduleScore(testFile)
    solved, score = Eval.eval_schedules(mySchedule)
    print 'Solved: %s %%' % solved                    
    print 'Score: %s %%  (%s / %s)' % (round(100*score/Eval.bestScore,4),score,Eval.bestScore)
