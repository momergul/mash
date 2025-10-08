#!/bash/bin

python -m setup.download_mash_data
mv mash_data/data data
mv mash_data/experiments experiments
rm -rf mash_data
