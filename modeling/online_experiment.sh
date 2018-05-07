outfolder=$1 #./tf_output/online_training/
infolder=$2 #./tf_output/official_training/

rm -r $outfolder
mkdir -p $outfolder
cp $2* $outfolder

# create asm batch
python create_ASM_batch.py 1000 1000 0.5 ../data False
# predict new data
python run_model.py pred ASM 1 16 True 1000 $outfolder new $outfolder 0
# train new data
python run_model.py train ASM 2 16 True 1000 $outfolder new $outfolder 1000

# create random batch
python create_ASM_batch.py 1000 1000 0.5 ../data True
# train old data
python run_model.py train ASM 2 16 True 1000 $outfolder old $outfolder 1000




for i in {2000..200000..1000}
  do
    # create asm batch
    python create_ASM_batch.py $i 1000 0.5 ../data False
    # predict new data
    python run_model.py pred ASM 1 16 True $i $outfolder new $outfolder $(($i-1000))
    # train new data
    python run_model.py train ASM 2 16 True $i $outfolder new $outfolder $i

    # create random batch
    python create_ASM_batch.py $i 1000 0.5 ../data True
    # train old data
    python run_model.py train ASM 2 16 True $i $outfolder old $outfolder $i

    # delete old files
    python end_batch.py $i $outfolder 2000
  done
