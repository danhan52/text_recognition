for i in {1000..100000..1000}
  do
  	python create_ASM_batch.py asm $i 1000 0.5 # create asm batch
  	# predict new data
  	python predict_model.py ASM 1 16 

  	# train new data

  	python create_ASM_batch.py rand $i 0.5 # create random batch
  	# train old data


  	python create_ASM_batch.py rem $i ./tf_output/estimator/ # delete old files
  done
