cd ..

echo "Removing old data and creating folders..."
rm -rf data/BenthamDataset
mkdir -p data/BenthamDataset

rm -rf data/iamHandwriting/ascii
mkdir -p data/iamHandwriting/ascii
rm -rf data/iamHandwriting/lines
mkdir -p data/iamHandwriting/lines
rm -rf data/iamHandwriting/Paritions
mkdir -p data/iamHandwriting/Partitions

echo "Unzipping Bentham data..."
mkdir data_raw/benth_temp
unzip -q data_raw/BenthamDatasetR0-GT.zip -d data_raw/benth_temp
mv data_raw/benth_temp/BenthamDatasetR0-GT/* data/BenthamDataset
rm -r data_raw/benth_temp

echo "Unzipping IAM handwriting data..."
tar -xzf data_raw/iamHandwriting/ascii.tgz -C data/iamHandwriting/ascii
tar -xzf data_raw/iamHandwriting/lines.tgz -C data/iamHandwriting/lines
unzip -q data_raw/iamHandwriting/largeWriterIndependentTextLineRecognitionTask.zip -d data/iamHandwriting/Partitions

cd bash_scripts