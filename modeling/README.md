## Text recognition modeling

To run one of the models as is, simply use the instructions found in the main README, copied below

### Running code
If you want to train a model on one of the existing datasets, run "bash train_and_validate.sh \<train\_location\>, \<test\_location\>, \<model\_location\>, \<epochs\>" from ~/modeling, where \<train\_location\> is the name of the folder in ~/data where the training data is found, \<test\_location\> is the name of the folder in ~/data where the test data is found, \<model\_location\> is the folder in ~/modeling/tf_output where the model should be saved to, and \<epochs\> is the number of epochs to run.

If you want to run the online learning process, run "bash online_experiment.sh \<output\_folder\> \<input\_folder\>" where \<output\_folder\> is the folder in ~/modeling/tf_output where the model should be saved to and \<input\_folder\> is the folder in ~/modeling/tf_output where the pre-trained model comes from.

If you want to run an experiment comparing the training of different file sizes, just run "bash file_size_experiment.sh". You can change which dataset and what file sizes you're using within the file.

The script end_batch.py is useful for printing out a lot of '\*'s at the end of a batch as well as cleaning up old files so that it doesn't get too cluttered.

### Modifying code - adding new models
The script run_model.py contains the code to set up a model and run it for a certain number of epochs. It isn't set up to do validation on it's own so you have to use train_and_validate.sh to do both parts. run_model.py sets up a model found in ~/modeling/models. run_bunch.py contains the code to actually run a specified number of epochs of either training or prediction on the data. This is in lieu of using the tensorflow estimator class because sometimes the CTC loss function will cause errors based on the labels. Rather than removing images entirely, I surround it with a try catch block and keep going when it fails. 

The folder ~/modeling/models currently contains only a CNN-RNN model but it would be easy to add another model by adding in a function that returns the appropriate values: a training operation, a loss function, character error rate, accuracy, raw probabilities, the actual words output from the model, and a prediction score.

The subfolder ~/modeling/models/model_builders contains the subcomponents for the models found in ~/modeling/models such as CNN and LSTM layers as well as functions for creating the loss function and iterator. The structure does not need to be strictly obeyed when building models but is there for modularity. Note that the files in ~/modeling/models/model_builders are named in a self-explanatory way to make it easy to find where pieces of the model came from.

Right now everything is in raw tensorflow because it's easier to set up CTC loss and more importantly the beam search decoder using tensorflow. That said, something that needs work before this can be fully integrated into Zooniverse and other platforms, not using the run_batch.py and run_model.py method of training and predicting the data. That could be fixed by figuring out why the CTC error occurs and using the tensorflow estimator class, it could also be done by converting the model to Keras. I would like to try the latter, but I am not yet sure that this is where time is best spent.

