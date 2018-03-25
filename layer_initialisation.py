import numpy as np
from forest_functions import GetTreeSplits, GetChildren


def InitFirstLayer(rf, strength01 = 1000.0):
    """
    Given a fitted random regression forest model rf, this function returns
    initialisation parameters for the first hidden layer of a neural net that
    mimics the random forest's prediction behaviour.
    - strength01 is a hyperparameter that determines how strongly the discrete
    step function is approximated.
    """

    # extract tree parameters
    trees, featurelist, threshlist = GetTreeSplits(rf)
    listcl, listcr = GetChildren(trees)

    # layer sizes
    HL1N = sum( [np.sum(tree.feature != -2) for tree in trees] )
    n_inputs = rf.n_features_

    # initialise first layer parameters
    W1 = np.zeros( [n_inputs, HL1N], dtype = 'float32')
    b1 = np.zeros([HL1N])
    b1 = np.array(b1, dtype = 'float32')


    currentnode = 0	# index of HL1 neuron to be assigned an input weight
    nodelist = []	# list (over all trees) of list of network neurons
                      # corresponding to tree splits

    # for every tree
    for i in range(len(trees)):
        cl = listcl[i]
        cr = listcr[i]

        currentsplit = 0  # index for current node while moving through tree i
        nodeCount = trees[i].node_count   # number of nodes in tree i
        nlist = np.zeros([nodeCount, 1])  # for each treenode save a neuronindex

        # go through all tree nodes in tree i
        while currentsplit < nodeCount:
            """
            Interpretation help:
            #active with (+1) if Wx>b 	<==> active (+1) when split to the right
            #active with (-1) if Wx<=b	<==> active (-1) when split to the left.
            """

            # What to do here: set weight and bias for current HL1 neuron
            if featurelist[i][currentsplit] != -2:   #not leaf node
                W1[featurelist[i][currentsplit], currentnode] = 1.0 * strength01
                b1[currentnode] = -threshlist[i][currentsplit] * strength01

                # store relationship from tree split index to HL1 neuron index
                nlist[currentsplit] = currentnode
                currentnode += 1

            # Where to go next: identify next node in tree (depth first)
            if cl[currentsplit] != -1: #( -1 means: empty child)
                currentsplit = cl[currentsplit]
            elif cr[currentsplit] != -1 :
                currentsplit = cr[currentsplit]
            else :
                currentsplit += 1
                #move to parent and next branch

        nodelist.append(nlist) #saving list of neurons corresponding to tree i.

    return W1, b1, nodelist




def GetTreePaths(trees):
    # List of lists containing the node indices for all paths through all trees
    jointsplitindlists = []
    # Litt of lists containing all path orders (left/right) through all trees
    jointsplitorderlists = []

    # lists of left and right children
    listcl, listcr = GetChildren(trees)

    for i in range(len(trees)):
        paths, orders = [], []
        cl = listcl[i].tolist()
        cr = listcr[i]

        leaf_nodes = np.where(cr == -1)[0].tolist()
        cr = cr.tolist()

        # for every leaf node get the path that led to it.
        for leaf in leaf_nodes:
            path, order = [], []
            c = leaf
            while c != 0:
                #find mother node of c
                if c in cl:
                    mother = cl.index(c)
                    direction = -1
                else:
                    mother = cr.index(c)
                    direction = +1
                c = mother
                path.append(c)
                order.append(direction)

            path.reverse()
            order.reverse()
            paths.append(path)
            orders.append(order)

        jointsplitindlists.append(paths)
        jointsplitorderlists.append(orders)

    return jointsplitindlists, jointsplitorderlists




#nodelist is from HL1
def InitSecondLayer(rf, nodelist, strength12=0.1,  L2param=0.8):
    """
    Given a fitted random regression forest model rf,
    a nodelist from the previous layer initialisation and hyperparameters,
    this function returns initialisation parameters for the second hidden layer
    of a neural net that mimics the random forest's prediction behaviour.
    """

    # extract tree paths
    trees = [rf.estimators_[i].tree_ for i in range(rf.n_estimators) ]
    jointsplitindlists, jointsplitorderlists = GetTreePaths(trees)

    # layer sizes
    HL1N = sum( [np.sum(tree.feature != -2) for tree in trees] )
    HL2N = sum( [np.sum(tree.feature == -2) for tree in trees] )

    # empty weight matrix and bias vector
    W2 = np.zeros([HL1N,HL2N], dtype = 'float32')
    b2 = np.zeros(HL2N, dtype = 'float32')

    fneurons = []   #the feature neuron in HL1 that occurs in a given path
    dneurons = []   #the path directions of the fneurons of a given path.
    leaf_neurons = []	#for storing indices of HL2-neurons belonging to leafs
    counter = 0

    # for each tree i
    for i in range(len(trees)):
        #get relevant split info
        indlist = jointsplitindlists[i]
        orderlist = jointsplitorderlists[i]
        neurons_used = []

        # identify split neurons in HL1 and their desired direction
        for k in range (len(indlist)):
            fneurons.append( nodelist[i] [ indlist[k] ] )
            dneurons.append( orderlist[k] )
            neurons_used.append(counter)
            counter +=1

        #append HL2-neuron-indices used in tree i.
        leaf_neurons.append(neurons_used)

        # set input weights and biases for second layer
        scndlayercount = 0
        for k in range(len(fneurons)): # for every feature neuron used (in HL1)
            inputns = fneurons[k]
            dirs = dneurons[k]
            for j in range(len(dirs)):
                W2[int(inputns[j]), scndlayercount] = dirs[j] * strength12
                b2[scndlayercount] = (-len(dirs)+0.5) * strength12
            scndlayercount +=1

    return W2, b2, leaf_neurons



def InitThirdLayer(rf, leaf_neurons):
    """
    Given a fitted random regression forest model rf,
    and a list of leaf_neurons from the previous layer initialisation,
    this function returns initialisation parameters for the third output layer
    of a neural net that mimics the random forest's prediction behaviour.
    """


    trees = [rf.estimators_[i].tree_ for i in range(rf.n_estimators) ]
    ntrees = rf.n_estimators

    # layer size
    HL2N = sum( [np.sum(tree.feature == -2) for tree in trees] )

    # empty weight vector. No bias, simply linear transformation.
    W3 = np.zeros([HL2N, 1], dtype = 'float32')

    # loop over trees
    for i in range(len(trees)):
        # identify the leaf node indices for tree i
        leaf_ind = np.where(trees[i].feature ==-2)

        # get the regression value for each of those leaves
        leaf_values = [e[0][0] for e in trees[i].value[leaf_ind].tolist()]

        # HL2 neurons corresponding to tree i
        tree_neurons = leaf_neurons[i]

        #compute  weights to output layer
        for k in range(len(leaf_values)):
            W3[tree_neurons[k]] = leaf_values[k] / float(ntrees) * 0.5

    return W3
