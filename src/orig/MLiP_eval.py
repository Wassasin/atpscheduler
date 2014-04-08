#! /usr/bin/env python
'''
Evaluation code for MLiP exercise 9.

Created on Mar 23, 2014

@author: Daniel Kuehlwein
'''

import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%% %(message)s',
                    datefmt='%d-%m %H:%M:%S')
logger = logging.getLogger('MLiP Setup')


class Problem():
    def __init__(self,dataString):
        tmp = dataString.strip().split('#')
        self.name = tmp[0]
        self.features = [float(x) for x in tmp[1].split(',')]
        self.times = [float(x) for x in tmp[2].split(',')]

class StrategyScheduleScore():
    """
    Class for computing the score of a strategy schedule.
    
    Example usage:
    Eval = StrategyScheduleScore('MLiP_train')
    Eval.eval_schedules('MLiP_train_example_schedule')

    """
    def __init__(self,dataFile):
        self.problems = {}
        self.strategyIndex = None
        with open(dataFile,'r') as IS:
            for line in IS:
                if self.strategyIndex == None:
                    tmp = line.split('#')[-1].strip()
                    tmp = tmp.split(',')
                    self.strategyIndex = {}
                    for i,n in enumerate(tmp):
                        self.strategyIndex[n] = i
                    continue
                pName = line.split('#')[0]
                self.problems[pName] = Problem(line)
        self.bestScore = 0.0
        self.totalSolved = 0
        for p in self.problems.itervalues():
            runTime = 300
            for i,time in enumerate(p.times):
                if time > -1 and time < runTime:
                    runTime = time
            if runTime < 300:
                self.totalSolved += 1
            self.bestScore += (300.0 - runTime)

    def eval_schedule(self,scheduleLine):
        """
        Evaluates a strategy schedule
        
        Strategy schedule syntax:
        problemName#strategyName:strategyTime,strategyName:strategyTime,..\n
        The sum of all strategyTimes must be <= 300
        
        Returns #ScheduleSolvesProblems, #TimeLeftComaparedToOptimalTime
        
        Example strategy schedules:
        Problems/CSR/CSR100+4.p#NewStrategy519:300
        Problems/SWV/SWV406+1.p#NewStrategy2452:10.1,NewStrategy3794:289.9
        """
        totalTime = 0.0
        tmp = scheduleLine.strip().split('#')
        pName = tmp[0]
        schedule = tmp[1].split(',')
        try:
            problem = self.problems[pName]
        except:
            logger.warning('No information about problem %s. Cannot evaluate' % pName)
            return False,0.0
        convertedSchedule = [x.split(':') for x in schedule]
        for sName,predTime in convertedSchedule:
            predTime = float(predTime)
            if totalTime + predTime > 300:
                return False,0.0            
            sI = self.strategyIndex[sName]
            realTime = problem.times[sI]
            if realTime > -1 and realTime < predTime:
                totalTime += realTime
                return True,300 - totalTime 
            else:
                totalTime += predTime
        return False,300 - totalTime
        
    def eval_schedules(self,scheduleFile):
        """
        Expects a file containing strategy schedules for all problems in the dataFile used during initialization.
        Evaluates the schedules and compares them with the optimal schedule.
        
        Returns the percentage of solved problems and the score.
        """
        score = 0.0
        solved = 0.0
        with open(scheduleFile,'r') as IS:
            for line in IS:
                pSolved,pScore = self.eval_schedule(line)
                score += pScore
                if pSolved:
                    solved += 1
        solvedPercentage = round(100*solved/self.totalSolved,2)
        return solvedPercentage,score
    
if __name__ == '__main__':
    trainData = sys.argv[1]
    scheduleFile = sys.argv[2]
    Eval = StrategyScheduleScore(trainData)
    solved, score = Eval.eval_schedules(scheduleFile)
    logger.info('Solved: %s %% of all solvable problems.' % solved)                    
    logger.info('Score: %s %%  (%s / %s)' % (round(100*score/Eval.bestScore,4),score,Eval.bestScore))
   
    