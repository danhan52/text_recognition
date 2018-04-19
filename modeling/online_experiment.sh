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
