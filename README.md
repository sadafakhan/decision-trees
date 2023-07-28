# decision-trees
```decision-trees``` builds a decision tree from the training tree, classifies the training and test data, and calculates the accuracy.

Args: 
* ```training_data```: vector file of training data in text format
* ```test_data```: vector file of test data in text format
* ```max_depth```: maximum depth of decision tree
* ```min_gain```: minimum information gain per level of decision tree 

Returns: 
* ```model_file```: decision tree produced by DT trainer (cf. model_ex)
* ```sys_output```: classification result on the trainingg and test data in the format "instanceName c1 p1 c2 p2..." (cf. sys_ex)
* ```acc_file```: confusion matrix and accuracy on the training and test data (cf. acc_ex)

To run: 
```
src/build_dt.sh input/train.vectors.txt input/test.vectors.txt max_depth min_gain output/model_file output/sys_output output/acc_file
```

NOTE: this implementation is not complete. 

HW2 OF LING572 (01/18/2022)