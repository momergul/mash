save_path=knowledge_sources

python -m verl_training.search.download --save_path $save_path

cat $save_path/part_* > $save_path/e5_Flat.index
rm $save_path/part_*

gzip -d $save_path/wiki-18.jsonl.gz
