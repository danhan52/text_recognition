mkdir -p ./tf_output/file_size

# do small images
cd ../preprocess
python preprocess_both_iam.py 0.25

cd ../modeling
mkdir -p ./tf_output/file_size/size25/
# training
python run_model.py train iamHandwriting 5 16 True 0 ./tf_output/file_size/size25/ new
# prediction
python run_model.py pred iamTest 1 16 True 1000 ./tf_output/file_size/size25/ new ./tf_output/file_size/size25/ 0


# do medium images
cd ../preprocess
python preprocess_both_iam.py 0.5

cd ../modeling
mkdir -p ./tf_output/file_size/size50/
# training
python run_model.py train iamHandwriting 5 16 True 0 ./tf_output/file_size/size50/ new
# prediction
python run_model.py pred iamTest 1 16 True 1000 ./tf_output/file_size/size50/ new ./tf_output/file_size/size50/ 0

# do full size images
cd ../preprocess
python preprocess_both_iam.py 1.0

cd ../modeling
mkdir -p ./tf_output/file_size/size100/
# training
python run_model.py train iamHandwriting 5 16 True 0 ./tf_output/file_size/size100/ new
# prediction
python run_model.py pred iamTest 1 16 True 1000 ./tf_output/file_size/size100/ new ./tf_output/file_size/size100/ 0
