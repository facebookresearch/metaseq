## Books3 Processing (in Spark)
Source/References:
* [sosek/bookscorpus](https://github.com/soskek/bookcorpus/issues/27#issuecomment-716104208)
* [HuggingFace datasets](https://github.com/huggingface/datasets/pull/856)

From the [sosek](https://github.com/soskek/bookcorpus/issues/27#issuecomment-716104208) download, filenames need to be normalized first before putting into HDFS to coalesce.  The following series of bash commands can be applied to clean up filenames:

```
First try:
[root@<host> Bibliotik]# hdfs dfs -put ./* /data/books3/raw/2022-09-18/
<-------- Showed failures writing to HDFS -------->
```
```
Local filename cleanup: 

find . -type f -iname "*[*" | while IFS= read -r line; do mv "$line" "$(printf %s "$line" | sed -re 's/(\[|\])//g')"; done;
find . -type f -iname "*(*" | while IFS= read -r line; do mv "$line" "$(printf %s "$line" | sed -re 's/(\(|\))//g')"; done;
find . -type f -iname "*:*" | while IFS= read -r line; do mv "$line" "$(printf %s "$line" | sed -re 's/\://g')"; done;
for f in */*,*; do mv -- "$f" "${f//,/}"; done
for f in */*\;*; do mv -- "$f" "${f//\;/}"; done
for f in */*\ *; do mv -- "$f" "${f//\ /}"; done
for f in */*-*; do mv -f "$f" "${f//-/}"; done
for f in */*\'*; do mv -f "$f" "${f//\'/}"; done
for f in */*_*; do mv -f "$f" "${f//_/}"; done
for f in */*@*; do mv -f "$f" "${f//@/}"; done
for f in */*\&*; do mv -f "$f" "${f//\&/}"; done
for f in */*.*.*.txt; do i="${f%.epub.txt}"; mv -f -- "$f" "${i//./}.epub.txt"; done
for f in */*; do mv -f "$f" "`echo $f | tr "[:upper:]" "[:lower:]"`"; done

for dir in */; do
cd $dir
for f in *; do mv -f "$f" "`echo $f | tr "[:upper:]" "[:lower:]"`"; done
cd ..
done

[root@<host> Bibliotik]# find . -type f | wc -l
196488
```

After the files are placed into HDFS, it is possible to extract the core contents of each book via `BooksHeaderCleaner.scala` and `BooksFooterCleaner.scala`, followed by applying the deduplication logic `BooksDedup.scala`.

### spark-shell settings
Code here can be executed directly in spark-shell with the following settings (can reduce `--num-executors` if needed):
```
spark-shell --master yarn --deploy-mode client \
--driver-memory 200g \
--executor-memory 42g \
--executor-cores 8 \
--num-executors 20 \
--conf spark.dynamicAllocation.enabled=False \
--conf spark.driver.cores=16 \
--conf spark.driver.maxResultSize=10g \
--conf spark.memory.offHeap.enabled=true \
--conf spark.memory.offHeap.size=8g \
--conf spark.memory.fraction=0.3 \
--conf spark.default.parallelism=2480 \
--conf spark.network.timeout=1200s \
--conf spark.files.maxPartitionBytes=268435456 \
--conf spark.files.openCostInBytes=134217728 \
--conf spark.locality.wait=20s \
--conf spark.task.cpus=2
```
