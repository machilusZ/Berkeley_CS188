# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
          return float('inf')
        elif successorGameState.isLose():
          return 0

        score = successorGameState.getScore()
        preFoodLeft = currentGameState.getFood().count(True)
        postFoodLeft = newFood.count(True)
        minGhostDistance = min([manhattanDistance(newPos, ghost.getPosition()) \
          for ghost in newGhostStates])

        foodDistMax = 0
        foodDistMin = 1000000
        foodDistTotal = 0
        for x in range(newFood.width):
          for y in range(newFood.height):
            if newFood[x][y]:
              pos = manhattanDistance(newPos, (x,y))
              foodDistTotal += pos
              foodDistMax = max(foodDistMax, pos)
              foodDistMin = min(foodDistMin, pos)

        score = score  + score * (preFoodLeft - postFoodLeft) +  \
              score / foodDistMin + score / (foodDistTotal + foodDistMax) 

        score  = score * score

        if minGhostDistance <= min(newScaredTimes) + 1:
          score = score / 5

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        
        numAgents = gameState.getNumAgents()

        def minMax(currentState, futureTurns, isMinAgent, agentIndex):
          if currentState.isWin() or currentState.isLose() or futureTurns <= 0:
            return self.evaluationFunction(currentState)
          legalMoves = currentState.getLegalActions(agentIndex)
          successorGameStates = [currentState.generateSuccessor(agentIndex, action) for action in legalMoves]
          nextAgent = (agentIndex + 1) % numAgents 
          scores = [minMax(state, futureTurns - 1, nextAgent != 0, nextAgent) for state in successorGameStates]
          return min(scores) if isMinAgent else max(scores)

        legalMoves = gameState.getLegalActions(0)
        successorGameStates = [gameState.generateSuccessor(self.index, action) for action in legalMoves] 
        scores = [minMax(state, self.depth * numAgents - 1, True, 1) for state in successorGameStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def isMin(agentIndex):
          return agentIndex % numAgents != 0

        def next(agentIndex):
          return (agentIndex + 1) % numAgents

        def nextIsMin(agent):
          return isMin(next(agent))

        def minAgent(state, depth, agent, alpha, beta):
          if depth <= 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state) 

          score = float('inf')
          legalMoves = state.getLegalActions(agent)
          if len(legalMoves) == 0:
            return 0
          for move in legalMoves:
            nextState = state.generateSuccessor(agent, move)
            if nextIsMin(agent):
              score = min(score, minAgent(nextState, depth - 1, next(agent), alpha, beta))
            else:
              score = min(score, maxAgent(nextState, depth - 1, next(agent), alpha, beta))
            if score < alpha:
              return score
            beta = min(beta, score)
          return score

        def maxAgent(state, depth, agent, alpha, beta):
          if depth <= 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

          score = float('-inf')
          legalMoves = state.getLegalActions(agent)
          if len(legalMoves) == 0:
            return 0
          for move in legalMoves:
            nextState = state.generateSuccessor(agent, move)
            if nextIsMin(agent):
              score = max(score, minAgent(nextState, depth - 1, next(agent), alpha, beta))
            else:
              score = max(score, maxAgent(nextState, depth - 1, next(agent), alpha, beta))
            if score > beta:
              return score
            alpha = max(alpha, score)
          return score

        bestScore = score = alpha = float('-inf')
        beta = float('inf')
        bestMove = None

        legalMoves = gameState.getLegalActions(self.index)
        for move in legalMoves:
          nextState = gameState.generateSuccessor(self.index, move)
          if nextIsMin(self.index):
            score = max(score, minAgent(nextState, numAgents * self.depth - 1, next(self.index), alpha, beta))
          else:
            score = max(score, maxAgent(nextState, numAgents * self.depth - 1, next(self.index), alpha, beta))
          if score > bestScore:
            bestScore = score
            bestMove = move
          alpha = max(alpha, score)

        return bestMove
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        numAgents = gameState.getNumAgents()

        def minMax(currentState, futureTurns, isMinAgent, agentIndex):
          if currentState.isWin() or currentState.isLose() or futureTurns <= 0:
            return self.evaluationFunction(currentState)
          legalMoves = currentState.getLegalActions(agentIndex)
          successorGameStates = [currentState.generateSuccessor(agentIndex, action) for action in legalMoves]
          nextAgent = (agentIndex + 1) % numAgents 
          scores = [minMax(state, futureTurns - 1, nextAgent != 0, nextAgent) for state in successorGameStates]
          if isMinAgent:
            return sum(scores) * 1.0 / len(legalMoves)
          else:
            return max(scores)

        legalMoves = gameState.getLegalActions(0)
        successorGameStates = [gameState.generateSuccessor(self.index, action) for action in legalMoves] 
        scores = [minMax(state, self.depth * numAgents - 1, True, 1) for state in successorGameStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:

      My algorithm promotes these features in states, oredered by priority:
      1. is a winning state
      2. eatting ghosts if possible
      3. eatting capsules
      4. do not get scared by ghosts if it's within the range of lifetime of capsules

      My algorithm tries to minimize these features in states, oredered by priority:
      1. distance from the closest ghosts
      2. distance from the closest food
      3. distance from the farthest food
      4. total distance from all food
      5. number of foods left
    """

    if currentGameState.isWin():
      return float('inf')
    elif currentGameState.isLose():
      return 0

    def distancesFromMe(lst, location, predicate, distance=manhattanDistance):
        return [distance(pos, location(x)) for x in lst if predicate(x)]
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    def intPos(pos):
      return (int(pos[0]), int(pos[1]))
    ghostDistances = distancesFromMe(ghostStates, lambda x: intPos(x.getPosition()), lambda x: True)
    foodDistances = distancesFromMe(currentGameState.getFood().asList(),\
               lambda x: x, lambda x: currentGameState.hasFood(x[0], x[1]))
    closestGhostDistance = min(ghostDistances)
    closestGhostIndex = ghostDistances.index(closestGhostDistance)
    score = score * 10 + score / (currentGameState.getNumFood() + 2 * sum(foodDistances)) \
            + score / (max(foodDistances) + 2 * min(foodDistances) + 4 * closestGhostDistance)
    if closestGhostDistance < scaredTimes[closestGhostIndex]:
      score /= 5
    if min(scaredTimes) != 0:
      score *= 3
    if closestGhostDistance <= 1 and scaredTimes[closestGhostIndex] > 1:
      score *= 15
    return score**2

# Abbreviation
better = betterEvaluationFunction

