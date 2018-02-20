echo "Unzipping Bentham data..."
mkdir ../data_raw/benth_temp
unzip -q ../data_raw/BenthamDatasetR0-GT.zip -d ../data_raw/benth_temp
mv ../data_raw/benth_temp/BenthamDatasetR0-GT/* ../data/BenthamDataset
rm -r ../data_raw/benth_temp

echo "Unzipping IAM handwriting data..."
tar -xzf ../data_raw/iamHandwriting/ascii.tgz -C ../data/iamHandwriting
tar -xzf ../data_raw/iamHandwriting/words.tgz -C ../data/iamHandwriting
tar -xzf ../data_raw/iamHandwriting/lines.tgz -C ../data/iamHandwriting