mkdir -p ./tf_output/file_size

# do small images
cd ../preprocess
python preprocess_both_iam.py 0.25

cd ../modeling
mkdir -p ./tf_output/file_size/size25/
bash train_and_validate_iam.sh ./tf_output/file_size/size25/

# do medium images
cd ../preprocess
python preprocess_both_iam.py 0.5

cd ../modeling
mkdir -p ./tf_output/file_size/size50/
bash train_and_validate_iam.sh ./tf_output/file_size/size50/

# do full size images
cd ../preprocess
python preprocess_both_iam.py 1.0

cd ../modeling
mkdir -p ./tf_output/file_size/size100/
bash train_and_validate_iam.sh ./tf_output/file_size/size100/