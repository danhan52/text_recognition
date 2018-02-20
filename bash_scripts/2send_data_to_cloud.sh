if [ $1 = 'gce' ] 
then
	echo "Transferring Bentham to google cloud..."
	gcloud compute scp ../data_raw/BenthamDatasetR0-GT.zip $2:~/text_recognition/data_raw

	echo "Transferring IAM to google cloud..."
	gcloud compute scp ../data_raw/iamHandwriting/ascii.tgz $2:~/text_recognition/data_raw/iamHandwriting
	gcloud compute scp ../data_raw/iamHandwriting/lines.tgz $2:~/text_recognition/data_raw/iamHandwriting
	gcloud compute scp ../data_raw/iamHandwriting/words.tgz $2:~/text_recognition/data_raw/iamHandwriting
elif [ $1 = 'aws' ]
then
	echo "Not implemented, ignore next messages"
	echo "Transferring Bentham to AWS..."
	echo "Transferring IAM to AWS..."
else
	echo "Must be gce or aws"
fi