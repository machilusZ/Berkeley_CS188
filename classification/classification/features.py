# features.py
# -----------
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


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractor(datum)

    def connected_components(datum):
        numRow = len(datum)
        numCol = len(datum[0])

        labels = []
        for i in range(numRow):
            labels.append([])
            for j in range(numCol):
                labels[i].append(0)


        equivalenceTable = dict()
        labelNum = 1

        datum = datum == 0

        def smallestNeighborLabel(arr, row, col):
            smallest = 0
            dr = [0, -1, 1]
            dc = [0, -1, 1]
            for c in dc:
                for r in dr:
                    if not (c == 0 and r == 0):
                        x = min(max(row + r, 0), numRow - 1)
                        y = min(max(col + c, 0), numCol - 1)
                        if arr[x][y] and labels[x][y] != 0:
                            if smallest == 0:
                                smallest = labels[x][y]
                            else:
                                smallest = min(labels[x][y], smallest)
            return smallest

        def associateNeighborLabel(arr, row, col, label, dictionary):
            dr = [0, -1, 1]
            dc = [0, -1, 1]
            for c in dc:
                for r in dr:
                    if not (c == 0 and r == 0):
                        x = row + r # min(max(row + r, 0), numRow - 1)
                        y = col + c # min(max(col + c, 0), numCol - 1)
                        if x >= 0 and x < numRow and y >= 0 and y < numCol:
                            if arr[x][y] and labels[x][y] != 0:
                                dictionary[label].append(labels[x][y])
                                dictionary[labels[x][y]].append(label)
            return dictionary                            

        for i in range(numRow):
            for j in range(numCol):
                if datum[i][j]:
                    newLabel = smallestNeighborLabel(datum, i, j)
                    if newLabel == 0:
                        labels[i][j] = labelNum
                        newLabel = labelNum
                        equivalenceTable[labelNum] = [labelNum]
                        labelNum += 1
                    else:
                        labels[i][j] = newLabel
                        associateNeighborLabel(datum, i, j, newLabel, equivalenceTable)

        finalLabels = dict()
        for key in equivalenceTable.keys():
            finalLabels[key] = min(equivalenceTable[key])

        return len(set(finalLabels.values()))

    numConnected = connected_components(datum)
    newFeatures = [0, 0, 0]
    if numConnected == 1:
        newFeatures[0] = 1
    elif numConnected == 2:
        newFeatures[1] = 1
    elif numConnected == 3:
        newFeatures[2] = 1
    features = np.array(features.tolist() + newFeatures) # features.tolist() + 
    return features


def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """


    # Put any code here...
    # Example of use:
    for i in range(len(trainPredictions)):
        prediction = trainPredictions[i]
        truth = trainLabels[i]
        if (prediction != truth):
            print "==================================="
            print "Mistake on example %d" % i
            print "Predicted %d; truth is %d" % (prediction, truth)
            print "Image: "
            print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
