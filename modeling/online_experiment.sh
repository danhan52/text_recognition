# create asm batch
python create_ASM_batch.py asm 0 1000 0.5
# predict new data
python predict_model.py ASM 1 16 ./tf_output/estimator/ 0
# train new data
python train_model.py ASM 2 16 ./tf_output/estimator/ new 0

# create random batch
python create_ASM_batch.py rand 1000 0.5
# train old data
python train_model.py ASM 2 16 ./tf_output/estimator/ old 0




for i in {1000..10000..1000}
  do
  	python create_ASM_batch.py asm $i 1000 0.5
	# predict new data
	python predict_model.py ASM 1 16 ./tf_output/estimator/ $i
	# train new data
	python train_model.py ASM 2 16 ./tf_output/estimator/ new $i

	# create random batch
	python create_ASM_batch.py rand $i 0.5
	# train old data
	python train_model.py ASM 2 16 ./tf_output/estimator/ old $i

	# delete old files
  	python create_ASM_batch.py rem $i ./tf_output/estimator/
  	python end_batch.py $i
  done





# file size ################################################################
mkdir -p ./tf_output/file_size

# do small images
cd ../preprocess
python preprocess_both_bentham.py 0.25

cd ../modeling
mkdir -p ./tf_output/file_size/size25/
bash train_and_validate_bentham.sh ./tf_output/file_size/size25/

# do medium images
cd ../preprocess
python preprocess_both_bentham.py 0.5

cd ../modeling
mkdir -p ./tf_output/file_size/size50/
bash train_and_validate_bentham.sh ./tf_output/file_size/size50/

# do full size images
cd ../preprocess
python preprocess_both_bentham.py 1.0

cd ../modeling
mkdir -p ./tf_output/file_size/size100/
bash train_and_validate_bentham.sh ./tf_output/file_size/size100/

# train and validate #################################################
# train model for one epoch
python run_model.py train BenthamDataset 1 16 False 0 $1 new
# prediction
python run_model.py pred BenthamDataset 1 16 False 0 $1 new $1 0
python end_batch.py 0

for i in {1000..4000..1000}
  do
	# train model for one epoch
	python run_model.py train BenthamDataset 1 16 False $i $1 new $1 $(($i-1000))
	# prediction
	python run_model.py pred BenthamDataset 1 16 False $i $1 new $1 $i

	# delete old files
	python end_batch.py $i $1 2000
  done