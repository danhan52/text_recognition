if [ $1 = 'gce' ] 
then
	echo "Transferring model to local..."
	gcloud compute scp $2:~/text_recognition/modeling/tf_output/estimator/* ../modeling/tf_output/estimator
	echo "Transferring graph to local..."
	gcloud compute scp $2:~/text_recognition/modeling/tf_output/graph/* ../modeling/tf_output/graph
	echo "Transferring predictions to local..."
	gcloud compute scp $2:~/text_recognition/modeling/tf_output/prediction/* ../modeling/tf_output/prediction
elif [ $1 = 'aws' ]
then
	echo "Transferring Bentham to AWS..."
	#scp -i $3 ../data_raw/BenthamDatasetR0-GT.zip $2:~/text_recognition/data_raw
	
	echo "Transferring IAM to AWS..."
	#scp -i $3 ../data_raw/iamHandwriting/ascii.tgz $2:~/text_recognition/data_raw/iamHandwriting
	#scp -i $3 ../data_raw/iamHandwriting/lines.tgz $2:~/text_recognition/data_raw/iamHandwriting
	#scp -i $3 ../data_raw/iamHandwriting/words.tgz $2:~/text_recognition/data_raw/iamHandwriting
else
	echo "Must be gce or aws"
fi
