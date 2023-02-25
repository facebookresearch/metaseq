// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import org.apache.spark.sql._
import org.apache.spark.sql.types._

class BooksDedup {

  def main(args: Array[String]): Unit = {

    if (args.length < 3) {
      System.err.println("Usage: <download date> <language>")
      System.exit(1)
    }

    // Download date, set to current date usually.
    val processDate = args(0) // "2022-09-18"
    // Arg that specifies base multiple of parallelism, commonly set to 160 or 320.
    val outputParallelism = args(1).toInt // 480
    val hdfsDir = "hdfs:///data/books3"

    // Initialize SparkSession and SparkContext
    val spark = SparkSession
      .builder
      .appName(s"Books3Dedup: ${processDate}")
      .getOrCreate()

    // read in from previously processed footer-stripped path
    val rawDat = spark.read.parquet(s"${hdfsDir}/processed_parquet/${processDate}/stage2_strip_footer/").repartition(outputParallelism)

    // (filename, text)
    val rawDatRdd = rawDat.rdd.map(row => (row.getAs[String](0), row.getAs[String](1)))

    // Step 1: remove identical duplicates, key on hashcode and size of file in case of collisions
    val fullHashDedupDat = rawDatRdd.map(x => ((x._2.hashCode, x._2.length), x)).reduceByKey((a, _) => a).map(x => (x._2, x._1._2))
    // fullHashDedupDat is now: ((filename, text), length_of_text)

    // Step 2: line by line dedup check
    // clean up lines per doc, sort by longest to shortest doc and assign index (lower index == longer doc)
    val docWithIdx = fullHashDedupDat.sortBy(_._2, ascending=false).zipWithIndex
    docWithIdx.persist() // (((filename, text), length_of_text), idx)

    val idxWithText = docWithIdx.map(x => (x._2, x._1._1._2)) // (idx, text)
    val hashLinesWithDocIdx = idxWithText.flatMap(x => {
      val idx = x._1
      val doc = x._2
      val docLines = doc.split("\n").map(_.trim()).filter(_.length > 50) // only compare lines sufficiently long
      docLines.map(line => ((line.hashCode, line.length), (idx, docLines.length)))
    })

    // compare on doc idx, lower doc idx (longer doc) is kept
    val dedupHashLines = hashLinesWithDocIdx.reduceByKey((a,b) => if (a._1 < b._1) a else b)

    // (idx, frac_remaining)
    val fracLinesRemaining = dedupHashLines.map(row => (row._2, 1)).reduceByKey(_+_).map(x => (x._1._1, x._2.toFloat/x._1._2))
    val booksToKeep = fracLinesRemaining.filter(_._2 >= 0.85)

    val dedupedBooksJoin = docWithIdx.map(x => (x._2, x._1._1)).join(booksToKeep)
    // pull out (filename, text)
    val dedupedBooks = dedupedBooksJoin.map(_._2._1).map(x => Row(x._1, x._2))
    val schema = new StructType()
      .add(StructField("filename", StringType, nullable = false))
      .add(StructField("text", StringType, nullable = false))
    val outputDf = spark.createDataFrame(dedupedBooks, schema)
    outputDf.repartition(32).write.mode(SaveMode.Overwrite).json(s"${hdfsDir}/processed_json/${processDate}/stage3_dedup/")
  }
}
