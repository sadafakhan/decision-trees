"""Sadaf Khan, LING572, HW2, 01/18/2022. Builds a decision tree from the training tree, classifies the training and
test data, and calculates the accuracy."""

# INITIALIZING
import os
import sys
import math
from collections import defaultdict

# vectors in text form
training_data = sys.argv[1]
test_data = sys.argv[2]

# determines when to stop building the DT
max_depth = int(sys.argv[3])
min_gain = int(sys.argv[4])

# output files
model_file = sys.argv[5]
sys_output = sys.argv[6]


# HELPER FUNCTIONS

# formats the data
def formatting(data):
    listified = open(os.path.join(os.path.dirname(__file__), data), 'r').read().split("\n")[:-1]
    formatted = []

    for line in listified:
        instance = []
        feats = set()
        split = line.split(" ")

        # add the class label to the instance
        instance.append(split[0])

        for item in split[1:-1]:
            feature = item.split(":")[0]
            feats.add(feature)

        # add dictionary of features to the instance
        instance.append(feats)
        formatted.append(instance)
    return formatted


# breaks apart the instances according to a breakpoint
def breaker(instances, bp):
    trues = []
    falses = []
    for instance in instances:
        # check that the breakpoint is in the feature vector
        if bp in instance[1]:
            trues.append(instance)
        else:
            falses.append(instance)
    return trues, falses


# keeps counts of number of labels per discrete label
def label_num(instances):
    labels = defaultdict(int)
    for instance in instances:
        label = instance[0]
        labels[label] += 1
    return labels


# calculates entropy of the given instances
def entropy(instances):
    labels = label_num(instances)
    total = sum(labels.values())

    ent_neg = 0
    for label in labels:
        p_ci = labels[label] / total
        i = p_ci * math.log2(p_ci)
        ent_neg += i

    return ent_neg * -1


# calculates info gain of a parent when split into its children
def infogain(parent, left, right):
    parent_entropy = entropy(parent)
    left_entropy = entropy(left)
    right_entropy = entropy(right)
    info_gain = parent_entropy - ((len(left) / len(parent)) * left_entropy) - (
            (len(right) / len(parent)) * right_entropy)
    return info_gain


# returns all possible breakpoints for a set of instances
def potential_bps(instances):
    lst = []
    for instance in instances:
        lst.append(instance[1])
    breakpoints = set().union(*lst)
    return breakpoints


# calculates info gain per breakpoint employed on the data and returns the breakpoint with the greatest info gain
def decider(instances):
    highest_gain = 0
    optimal_bp = None
    breakpoints = potential_bps(instances)

    # iterate through the potential splits
    for bp in breakpoints:
        trues, falses = breaker(instances, bp)

        # if the split results in a wholesale import of one set into one child, pass
        if len(trues) == 0 or len(falses) == 0:
            continue

        # check if this the best split so far
        ig = infogain(instances, trues, falses)
        if ig >= highest_gain:
            highest_gain = ig
            optimal_bp = bp

    return highest_gain, optimal_bp


# turns dictionary of distributions into strings
def probs_to_str(distribution):
    s = ""
    for label in distribution:
        s += label + " " + str(distribution[label]/sum(distribution.values())) + " "
    return s

# CLASSES
class Leaf:
    """A terminal node. Has a dictionary mapping class label to the number of times an instance of that label reaches
    this leaf.
    """
    def __init__(self, instances):
        self.classLabel = label_num(instances)


class DecisionNode:
    """Holds a reference to a breakpoint and the two resultant child nodes.
    """
    def __init__(self, bp, left, right):
        self.bp = bp
        self.left = left
        self.right = right


# EXECUTION

# builds the tree using all the helper functions.
def builder(instances, depth):
    ig, bp = decider(instances)

    # recursive case; as long as ig is high enough and tree is short enough, keep splitting
    if ig >= min_gain & depth < max_depth:
        depth += 1
        trues, falses = breaker(instances, bp)

        # build the branches
        true_side = builder(trues, depth)
        false_side = builder(falses, depth)
        return DecisionNode(bp, true_side, false_side)

    # base case; shouldn't split further
    else:
        return Leaf(instances)


# classifies a test instance by running it through the tree
def DTclassifier(test_instance, node):
    # if we've reached the end of the tree, return the node and the predicted class
    if isinstance(node, Leaf):
        return node, node.classLabel

    # otherwise, choose a direction as we go down the tree
    if node.bp in test_instance[1]:
        return DTclassifier(test_instance, node.left)
    else:
        return DTclassifier(test_instance, node.right)


# converts the tree model into text
def model_lines(node, s=""):
    if isinstance(node, Leaf):
        return s + " " + str(sum(node.classLabel.values())) + " " + probs_to_str(node.classLabel) + "\n"
    else:
        return model_lines(node.right, s + "&!" + node.bp) + model_lines(node.left, s + "&" + node.bp)


training_vec = formatting(training_data)
testing_vec = formatting(test_data)
tree = builder(training_vec, 0)

# model produced by the DT trainer
with open(model_file, 'w') as model:
    for instance in training_vec:
        model.write(model_lines(tree) + "\n")


# classification result of the training and test data
with open(sys_output, 'w') as sysoutput:
    sysoutput.write("%%%%% training data:\n")

    for i in range(len(training_vec)):
        sysoutput.write("array:" + str(i) + " ")
        leaf, class_label = DTclassifier(training_vec[i], tree)
        sysoutput.write(probs_to_str(class_label) + "\n")

    sysoutput.write("%%%%% testing data:\n")
    for j in range(len(testing_vec)):
        sysoutput.write("array:" + str(j) + " ")
        leaf, class_label = DTclassifier(testing_vec[j], tree)
        sysoutput.write(probs_to_str(class_label) + "\n")