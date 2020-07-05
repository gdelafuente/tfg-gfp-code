mkdir datasets

echo "Downloading Places365..."
wget http://data.csail.mit.edu/places/places365/val_256.tar
tar -xvf val_256.tar
rm val_256.tar
mv val_256 datasets/places
wget http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -xvf filelist_places365-standard.tar
rm filelist_places365-standard.tar
rm places365_test.txt
rm places365_train_standard.txt
mv categories_places365.txt datasets/places
mv places365_val.txt datasets/places
echo "Places downloaded."

echo "Downloading Indoor Scene Recognition..."
wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
tar -xvf indoorCVPR_09.tar
rm indoorCVPR_09.tar
mv Images datasets/indoor
echo "Indoor downloaded."

echo "Downloading SUN Database..."
wget https://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
tar -xvf SUN397.tar.gz
rm SUN397.tar.gz
mv SUN397 datasets/sun
rm datasets/sun/*.txt
echo "SUN downloaded."