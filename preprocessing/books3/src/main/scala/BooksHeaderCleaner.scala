// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import org.apache.log4j.{Level, LogManager}

import org.apache.spark.sql._
import org.apache.spark.sql.types._

class BooksHeaderCleaner {

  def digitClean(st: String): Option[Int] = {
    if (st(0).isDigit) {
      val isDigitChar = st.map(_.isDigit)
      val lastDigit = isDigitChar.indexOf(false)
      try {
        if (lastDigit == -1) { // whole thing is a number
          Some(st.toInt)
        } else {
          Some(st.slice(0, lastDigit).toInt)
        }
      }
      catch { // swallow any exception
        case _: Throwable => None
      }
    }
    else {
      None
    }
  }

  def maybeCookbook(digitLineLoc: Array[(Int, Int)]): Boolean = {
    // if 1. 2. are relatively similar in count and there's more than 5 occurrences of each
    // this might be a cookbooks and not a table of contents lol
    val digitCounts = digitLineLoc.groupBy(_._1).mapValues(_.length)
    val numOnes = digitCounts.getOrElse(1, -1)
    val numTwos = digitCounts.getOrElse(2, -1)
    val maxCount = numOnes.max(numTwos)
    numOnes >= 3 && numTwos >= 3 && (numOnes - numTwos).abs <= 0.1 * maxCount
  }

  // call into this when > 50% of lines start with digits
  // maxDigitStripped is Some(int) with int of max digit previously stripped, or none if no stripping has happened
  // returns (filtered lines, whether or not filtering happened)
  def processDigits(allLines: Array[String], topLineInd: Int, maxDigitStripped: Option[Int]): (Array[String], Boolean, Option[Int]) = {
    val topLines = allLines.slice(0, topLineInd)
    // grab max number and which line (maxDigitLoc) this max number shows up at
    // (digit, line number)
    val digitLineLoc = topLines.map(l => digitClean(l.trim())).zipWithIndex.flatMap(rec => {
      rec._1 match {
        case Some(d) => Some(d, rec._2)
        case _ => None
      }
    })
    val (maxDigit, maxDigitLoc) = digitLineLoc.maxBy(_._1)
    val shouldStrip = if (maxDigitStripped.isDefined) {
      // have stripped digit before, requires current max to be greater than max stripped
      // in order to continue stripping
      maxDigit > maxDigitStripped.get
    }
    else {
      // haven't stripped before, need maxDigit > 5 (might truncate first 5 steps of a single recipe in a cookbook, oh well)
      maxDigit > 5
    }

    val isCookbook = maybeCookbook(digitLineLoc)

    if (shouldStrip && !isCookbook) {
      (allLines.slice(maxDigitLoc + 1, allLines.length), true, Some(maxDigit))
    } else {
      (allLines, false, maxDigitStripped)
    }
  }

  // recursively tries to strip header
  // alreadyStrippedHeader is true if we've already matched on a header and want to check for table of contents
  // maxDigitStripped contains Some(int) if we're in the process of stripping digits
  // output: (loopCount, fileName, book content, number of lines in book)
  def tryStripHeader(fileName: String, allLines: Array[String], filterFrac: Double, loopCount: Int,
                     alreadyStrippedHeader: Boolean, maxDigitStripped: Option[Int]): (Int, String, String, Int) = {
    val topLineInd = (allLines.length * filterFrac).toInt
    val loopLimit = 10
    if (topLineInd < 5 || loopCount > loopLimit) { // stop looping, might be stripping too much
      (loopCount, fileName, allLines.mkString("\n"), allLines.length)
    }
    else {
      val topLines = allLines.slice(0, topLineInd).map(_.toLowerCase().trim()) // used for filtering logic only

      // digit check for page / table of contents
      val linesStartingWithDigit = topLines.map(l => l(0).isDigit) // list of bools, mapping to whether line starts with digit or not
      val numLinesStartingWithDigit = linesStartingWithDigit.count(_ == true)
      if (numLinesStartingWithDigit * 2 > topLines.length) {
        // drop lines that start with digits if max digit in this group is > 5
        try {
          val (filteredDigitLines, isFiltered, newMaxDigitStripped) = processDigits(allLines, topLineInd, maxDigitStripped)
          if (isFiltered) { // keep trying to filter more digit lines
            tryStripHeader(fileName, filteredDigitLines, filterFrac, loopCount + 1, alreadyStrippedHeader, newMaxDigitStripped)
          } else { // likely was a cookbook or some other thing that failed filtering logic, despite all the digits, leave alone for now
            (loopCount, fileName, allLines.mkString("\n"), allLines.length)
          }
        }
        catch { // swallow any exception
          case _: Throwable => (-1, fileName, "", 0)
        }
      }
      else { // no digits-like table of contents present, check for other headers
        if (alreadyStrippedHeader) {
          // already stripped out a header before, no need to do again
          (loopCount, fileName, allLines.mkString("\n"), allLines.length)
        }
        else {
          // haven't stripped out a header-link chunk yet, time to check for any within this chunk
          val headerMatchLine = topLines.map(l => {
            val wordList = l.split(" ").map(_.trim()).filter(_.nonEmpty)

            // requires lines to be <= 30 chars to match
            val shortStringList = List("acknowledgments", "introduction", "contents", "title", "prologue", "foreword")

            // requires lines to be <= 400 chars to match
            val longStringList = List("all rights reserved", "copyright", "©", "already a subscriber",
              "thank you for downloading", "mcgraw-hill education", "about the author", "by the same author",
              "simon & schuster", "®", "tel: ", "penguin books", "harpercollins", "@", "author's note",
              "prior written permission", "about the publisher", "penguin group", "www.penguin", "penguingroup.com")

            // filter for whether or not line contains any string from the above two lists
            val stringMatchFilter = (l.length <= 30 && shortStringList.exists(s => l.contains(s))) || (l.length <= 400 && longStringList.exists(s => l.contains(s)))
            // check if there's any isbn identifiers
            val isbnFilter = if (l.length <= 400) wordList.exists(_.contains("isbn")) else false
            // check if there's mention of this being an e-book
            val ebookFilter = if (l.length <= 30) wordList.exists(w => (w.contains("ebook") || w.contains("e-book")) && !(w.contains("notebook") || w.contains("note-book") || w.contains("facebook") || w.contains("casebook"))) else false
            // misc.
            val additionalFilter = if (l.length <= 100) l.startsWith("www.")  || l.startsWith("introduction") else false

            stringMatchFilter || isbnFilter || ebookFilter || additionalFilter
          })
          val lastInd = headerMatchLine.toList.lastIndexWhere(_ == true)

          if (lastInd >= 0) {
            // strip up to where header match occurred and continue stripping one more time
            tryStripHeader(fileName, allLines.drop(lastInd + 1), filterFrac, loopCount + 1, alreadyStrippedHeader=true, maxDigitStripped=maxDigitStripped)
          }
          else {
            // no header-like line found, just return
            (loopCount, fileName, allLines.mkString("\n"), allLines.length)
          }
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    if (args.length < 2) {
      System.err.println("Usage: <download date> <file_parallelism>")
      System.exit(1)
    }

    // Download date, set to current date usually.
    val processDate = args(0) // "2022-09-18"

    // Arg that specifies base multiple of parallelism, commonly set to 160 or 320.
    val outputParallelism = args(1).toInt // 480

    val hdfsDir = "hdfs:///data/books3"
    val outputParquetPath = s"${hdfsDir}/raw_parquet/${processDate}/"

    // Initialize SparkSession and SparkContext
    val spark = SparkSession
      .builder
      .appName(s"Books3HeaderCleaner: ${processDate}")
      .getOrCreate()
    val sc = spark.sparkContext

    // Save all the books files into parquet (reduce file count to ~240 instead of ~196k)
    // Read in with filename via sc.wholeTextFiles
    val rawDat = sc.wholeTextFiles(path = s"${hdfsDir}/raw/${processDate}/*/*", outputParallelism)

    val schema = new StructType()
      .add(StructField("filename", StringType, nullable = false))
      .add(StructField("text", StringType, nullable = false))

    val rowRdd = rawDat.map(x => Row(x._1, x._2))
    val docsDf = spark.createDataFrame(rowRdd, schema)
    docsDf.repartition(outputParallelism/2).write.mode(SaveMode.Overwrite).parquet(outputParquetPath)

    // Process header stripping

    // *** Setup for debugging ***
    // val outputParallelism = 480
    // val hdfsDir = "hdfs:///data/books3"
    // val outputParquetPath = "hdfs:///data/books3/raw_parquet/2022-09-18/"

    val rawParquetDat = spark.read.parquet(outputParquetPath).repartition(outputParallelism)
    // (filepath, book content)
    val rawParquetDatRdd = rawParquetDat.rdd.map(row => (row.getAs[String](0), row.getAs[String](1)))
    // split book on newlines, remove lines that are empty when trimmed, and filter for books with more than 10 lines
    val removeEmptyLinesRdd = rawParquetDatRdd.map(x => (x._1, x._2.split("\n").filter(_.trim().nonEmpty))).filter(_._2.length > 10)
    // process each book to try and remove header, filtering out books that have < 100 lines after stripping
    // Debug info: tally of (loopCounts, numbooks) after this line:
    //  Array((0,10967), (1,179794), (2,1753), (3,493), (4,111), (5,32), (6,15), (7,3), (8,4), (9,2), (11,23))
    val tryStripHeaderRdd = removeEmptyLinesRdd.map(x => tryStripHeader(x._1, x._2, filterFrac=0.1, loopCount=0, alreadyStrippedHeader=false, maxDigitStripped=None)).filter(_._4 > 100)

    val countFailures = tryStripHeaderRdd.filter(_._1 == -1).count()
    log.warn(s"Found ${countFailures} parse failures!")
    // filter out all the books that have stripped over 10 times since those seem to mostly be empty books
    // e.g. FromHell.epub.txt
    val successfulParse = tryStripHeaderRdd.filter(l => l._1 > -1 && l._1 <= 10)

    val schemaWithHeadStrip = new StructType()
      .add(StructField("filename", StringType, nullable = false))
      .add(StructField("text", StringType, nullable = false))
      .add(StructField("header_strip_count", IntegerType, nullable = false))

    val outputRdd = successfulParse.map(x => Row(x._2, x._3, x._1))
    val outputDf = spark.createDataFrame(outputRdd, schemaWithHeadStrip)

    outputDf.repartition(outputParallelism/2).write.mode(SaveMode.Overwrite)
      .parquet(s"${hdfsDir}/processed_parquet/${processDate}/stage1_strip_header/")
    outputDf.repartition(outputParallelism/2).write.mode(SaveMode.Overwrite)
      .json(s"${hdfsDir}/processed_json/${processDate}/stage1_strip_header/")

    // End the Spark session.
    spark.stop()
  }
}
