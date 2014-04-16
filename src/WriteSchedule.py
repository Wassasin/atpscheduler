from StrategyScheduler import StrategyScheduler
from MLiP_eval import StrategyScheduleScore

if __name__ == '__main__':
    trainFile = 'orig/MLiP_train' 
    testFile = 'orig/MLiP_train'
    mySchedule = 'My_MLiP_train_example_schedule'
    
    SS = StrategyScheduler()
    SS.fit_file(trainFile)
    
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
