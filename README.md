# Text Recognition for Zooniverse

The goal of this repository is to use and create methods for handwritten text recognition and incorporate them into the Zooniverse platform.

Some overall notes for the repository:  
While it's technically not the home directory, I'll be refering to the top level directory as "~" for convenient notation.  
I used python 3.6 within Anaconda 4.4 to run everything. The neural networks were run on tensorflow 1.4.

## Data preparation
To run everything in the repository, first you need to do some set up.

0. While feasible to do certain parts of this on your local machine, the main tensorflow models are very computationally expensive and prohibitive to use on a normal computer without a GPU. With that in mind, the following steps don't actually differ depending on where you're doing this all, just that there will be some set up that's expected. 
1. From a command line, navigate to the folder ~/bash_scripts and run "bash 1setup_folders.sh". This creates a couple of necessary folders.
2. Download the data files - if using a cloud service, upload to the cloud.
    1. From the IAM Handwriting Database, download the zipped ascii.tgz and lines.tgz files from [this page](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database). You have to make an account first, but it's free. You'll also need the training-validation splits found in the Tasks section of [this page](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). Put these into the folder ~/data_raw/iamHandwriting
    2. From tranScriptorium, download the [Bentham Collection data](http://transcriptorium.eu/datasets/bentham-collection/). You specifically want the "Ground Truth" zip file. Put this into the folder ~/data_raw
    3. To do the online training stuff as it currently is, you need data from the [Anti-Slavery Manuscripts Zooniverse project](https://www.antislaverymanuscripts.org). If you're a random person who stumbled upon this repository, you're not allowed to access this data. If you're a Zooniverse researcher, ask Sam for the data, and she'll set you up. You specifically need a classifications export and a subject export and you should put the data into ~/data_raw/ASM
3. Navigate to ~/bash_scripts and run "bash 2decompress_data.sh". This does require the unzip command which doesn't come pre-installed on all Linux systems so if it doesn't work, you'll want to do "sudo apt-get install unzip". Two notes for this step: 1) This deletes the previously existing data in the IAM and Bentham folders. 2) The decompressing isn't verbose because there are a few thousand files and that would slow down an already slow process.
4. Now you can preprocess the data. If you want the easy route, navigate to ~/preprocess and run "python preprocess_all.py" and wait a while (like an hour) because the preprocessing, specifically for the Anti-Slavery Manuscripts stuff, is slow. If you want to do something else, go into the ~/preprocess folder and look at the README found there. At this point, the data is ready for ingestion into the model.

## Modeling
There is a lot more detail in the README found in the modeling folder, but what follows is a basic description of how to run the models.

If you want to train a model on one of the existing datasets, run "bash train_and_validate.sh \<train\_location\>, \<test\_location\>, \<model\_location\>, \<epochs\>" from ~/modeling, where \<train\_location\> is the name of the folder in ~/data where the training data is found, \<test\_location\> is the name of the folder in ~/data where the test data is found, \<model\_location\> is the folder in ~/modeling/tf_output where the model should be saved to, and \<epochs\> is the number of epochs to run.

If you want to run the online learning process, run "bash online_experiment.sh \<output\_folder\> \<input\_folder\>" where \<output\_folder\> is the folder in ~/modeling/tf_output where the model should be saved to and \<input\_folder\> is the folder in ~/modeling/tf_output where the pre-trained model comes from.

The main architecture used in this model is based on the model in the repository:
https://github.com/solivr/tf-crnn
which in turn is based upon the work in
https://github.com/bgshih/crnn

## Data analysis
Before running anything in ~/results, you must copy the .csv files output by the training process to ~/results/data.

Everything in the ~/results folder is not as automated as the rest of the code, so in order to use it, you'll need to look at the code (it's well commented) and modify as you see fit. The file training_validation_plots.R shows how to make plots from the data output by the modeling. The file online_experiment_results.ipynb shows how to get examples of the transcriptions alongside the images and error rates. 
