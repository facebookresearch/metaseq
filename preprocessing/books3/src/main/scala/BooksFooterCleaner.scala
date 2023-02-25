// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import scala.collection.mutable

import org.apache.spark.sql._
import org.apache.spark.sql.types._

class BooksFooterCleaner {

  // passed in reversed doc as allLines, filterFrac is set to fraction of lines to consider stripping in each pass
  // and loopCount tracks the number of passes we've stripped so far from a given doc
  // returns (number of passes, fileName, stripped doc, strings matched on & count of matches)
  def tryStripFooter(fileName: String, allLines: Array[String], filterFrac: Double, loopCount: Int, matchedCounts: mutable.Map[String, Int]): (Int, String, String, mutable.Map[String, Int])= {
    val numLines = allLines.length
    val fracInd = (numLines * filterFrac).toInt

    // if we only have < 10 lines to consider or have stripped 10 times already, return the below & stop stripping
    if (fracInd < 10 || loopCount > 20) {
      (loopCount, fileName, allLines.reverse.mkString("\n"), matchedCounts)
    }
    else {
      // take top frac of lines to process
      val topLines = allLines.slice(0, fracInd).map(_.toLowerCase().trim())
      // requires line length to be <= 30 to match on containing any of the below strings
      val shortStringList = List("acknowledgments", "acknowledgements", "@", "author's note", "index", "glossary",
        "bibliography", "contributors", "other works", "further reading", "leave a review", "notes", "the end",
        "resources", "works cited", "random house")
      // requires line length to be <= 200 to match on containing any of the below strings
      val longStringList = List("all rights reserved", "copyright", "©", "already a subscriber",
        "thank you for downloading", "mcgraw-hill education", "about the author", "note on the author",
        "by the same author", "simon & schuster", "®", "tel: ", "penguin books", "harpercollins",
        "thank you for purchasing")

      val stringMatchList = topLines.map(x => {
        val matchRes = x match {
          case l if l.length <= 30 && shortStringList.exists(s => l.contains(s)) => shortStringList.find(s => l.contains(s)).getOrElse("")
          case l if l.length <= 200 && longStringList.exists(s => l.contains(s)) => longStringList.find(s => l.contains(s)).getOrElse("")
          case l if l.split(" ").exists(_.contains("isbn")) => "isbn"
          case l if l.contains("www.") => "www."
          case l if l.startsWith("copyright") => "copyright"
          case _ => ""
        }
        matchRes
      })
      val lastStringMatchInd = stringMatchList.toList.lastIndexWhere(_.nonEmpty)
      if (lastStringMatchInd >= 0) {
        val matchedString = stringMatchList(lastStringMatchInd)
        matchedCounts(matchedString) += 1
        if (matchedCounts(matchedString) > 3) { // return - TODO: continue stripping special chars?
          (loopCount, fileName, allLines.reverse.mkString("\n"), matchedCounts)
        }
        else {
          tryStripFooter(fileName, allLines.drop(lastStringMatchInd + 1), filterFrac, loopCount + 1, matchedCounts)
        }
      }
      else {
        // failed above string match footer stripping, need to do special char / index check
        // truncation will be at min of these sections that are explicitly "protected" (quiz, ingredients, etc.)

        // save first occurrence of anything that resembles q&a at end of book
        val quizMatchList = topLines.map(l => {
          val wordSet = l.split(" ").map(_.trim()).filter(_.nonEmpty).toSet
          wordSet.contains("answer:")
        })
        val firstQuizInd = quizMatchList.toList.indexOf(true)

        // save first occurrence of anything that resembles potential for a cookbook
        val ingredientsMatchList = topLines.map(l => {
          if (l.length < 100) {
            val wordSet = l.split(" ").map(_.trim()).filter(_.nonEmpty).toSet
            wordSet.contains("ingredients") || wordSet.contains("instructions") || wordSet.contains("recipe") || wordSet.contains("•") || wordSet.contains("kcals")
          } else {
            false
          }
        })
        val firstIngredientInd = ingredientsMatchList.toList.indexOf(true)

        val chapterWords = List("chapter", "part", "section", "page")
        // look for indices of lines that match the below special char logic
        val specialCharMatchList = topLines.map(l => {
          val wordListSpace = l.split(" ").map(_.trim()).filter(_.nonEmpty)
          val wordListComma = l.split(",").map(_.trim()).filter(_.nonEmpty)
          val wordList = if (wordListSpace.length < wordListComma.length) wordListSpace else wordListComma
          val numWords = wordList.length

          // not including pipes and dashes here, since they're sometimes used for table formatting: ---|---|--- etc.
          // not including period counting, since they show up in plays (delimiter for speaker / text) and code
          val numCommas = l.count(_ == ',')
          val numUnderscores = l.count(_ == '_')
          val numHashes = l.count(_ == '#')
          val excludeFromSpecialCharCheck = l.length < 10 || l(0) == '"' || !chapterWords.exists(x => l.contains(x))
          val specialCharFilter = if (!excludeFromSpecialCharCheck)(3*numUnderscores + 3*numHashes) > l.length else false

          val commaCheck = if (numCommas > 0) numCommas > numWords else false

          commaCheck || specialCharFilter
        })

        val pageCheck = topLines.map(l => {
          val wordList = l.split(" ").map(_.trim()).filter(_.nonEmpty)
          val numWords = wordList.length
          // filter out lines that end with numbers, has between (2, 20) "words", is not a section/chapter header, and is not a year
          // goal is to grab "index-like" lines that reference pages in the book
          val lastWord = wordList.last
          val numDigitsInLastWord = lastWord.count(_.isDigit)
          val hasDigitsInLastWord = 2 * numDigitsInLastWord > lastWord.length
          // avoid filtering lines that look like citations, or references to historical works / art
          val yearCheck = if (numWords > 2) lastWord.length >= 4 && lastWord.take(4).count(_.isDigit) == 4 && (lastWord.startsWith("1") || lastWord.startsWith("2")) else false
          // citation filter for lines like: "> (Hume [1757] 1993c: 126)" or "(Marx 1973: 549)"
          val citationFilter = l.contains("(") && l.contains(")") && l.contains(":")
          !citationFilter && !yearCheck && hasDigitsInLastWord && numWords > 2 && numWords < 20 && !chapterWords.exists(x => l.contains(x))
        })
        val pageCheckWithLineIdx = pageCheck.zipWithIndex // (true/false, line_idx)
        val consecutivePageCheck = pageCheckWithLineIdx.zip(pageCheckWithLineIdx.tail).filter(x => x._1._1 && x._2._1).map(_._2) // this is a reversed doc
        val numConsecutivePages = consecutivePageCheck.count(_._1 == true)
        val shouldFilterPages = numConsecutivePages >= 0.25 * topLines.length

        val specialCharAndPageMatchList = if (shouldFilterPages) specialCharMatchList.zip(pageCheck).map(x => x._1 || x._2) else specialCharMatchList

        // TODO: add check for whether or not first letter is in reverse alphabetical ordering (docs are reversed)

        val hasProtectedSection = firstQuizInd >= 0 || firstIngredientInd >= 0
        val specialCharInd = if (hasProtectedSection) {
          val minProtectedInd = firstQuizInd.min(firstIngredientInd)
          // grab the last special char ind line that is before a protected section
          specialCharAndPageMatchList.toList.slice(0, minProtectedInd).lastIndexWhere(_ == true)
        } else {
          specialCharAndPageMatchList.toList.lastIndexWhere(_ == true)
        }

        if (specialCharInd >= 0) {
          if (hasProtectedSection) {
            // if we've hit a chunk with a protected section and special chars, just drop what we can and return
            (loopCount, fileName, allLines.drop(specialCharInd + 1).reverse.mkString("\n"), matchedCounts)
          }
          else { // continuing filtering for special chars
            tryStripFooter(fileName, allLines.drop(specialCharInd + 1), filterFrac, loopCount + 1, matchedCounts)
          }
        }
        else { // no special chars, nor footer match
          if (loopCount < 3 && matchedCounts.isEmpty) { // expand search
            tryStripFooter(fileName, allLines, filterFrac * 2, loopCount + 1, matchedCounts)
          }
          else {
            (loopCount, fileName, allLines.reverse.mkString("\n"), matchedCounts)
          }
        }
      }
    }
  }

  // inputLine contains whitespaces here
  def tryCleanLine(inputLine: String, cleanCount: Int): String = {
    if (cleanCount > 10) {
      inputLine
    } else {
      val wordList = inputLine.split(" ").map(_.trim()).filter(_.nonEmpty)
      val wordSet = wordList.toSet
      val trimmedInput = inputLine.trim()
      val cleanedLine = trimmedInput match {
        case l if l.startsWith("#### ") => inputLine.replace("#### ", "")
        case l if l.endsWith(" ####") => inputLine.replace(" ####", "")
        case l if l.startsWith("### ") => inputLine.replace("### ", "")
        case l if l.endsWith(" ###") => inputLine.replace(" ###", "")
        case l if l.startsWith("## ") => inputLine.replace("## ", "")
        case l if l.endsWith(" ##") => inputLine.replace(" ##", "")
        case _ if wordSet.size == 1 && wordList(0).length == 1 && wordList.length > 2 => "" // line consists of just one character, repeated more than twice
        case l if l.startsWith("**") && l.endsWith("**") && !l.slice(2, l.length - 2).contains("**") => inputLine.replace("**", "")
        case l if l.startsWith("_") && l.endsWith("_") && !l.slice(1, l.length - 1).contains("_") => inputLine.replace("_", "")
        case _ => inputLine
      }

      if (cleanedLine != inputLine) { // continue trying to clean if we cleaned this round
        tryCleanLine(cleanedLine, cleanCount + 1)
      }
      else { // didn't do anything this round, just return input
        inputLine
      }
    }
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: <download date> <file_parallelism>")
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
      .appName(s"Books3FooterCleaner: ${processDate}")
      .getOrCreate()

    // read in from previously processed header-stripped path
    val rawDat = spark.read.parquet(s"${hdfsDir}/processed_parquet/${processDate}/stage1_strip_header/").repartition(outputParallelism)
    // split books on newlines and reverse for footer processing
    // input is (filename, book contents, number of times header stripping occurred)
    // TODO: potentially filter out those where there was too much header stripping already?
    val rawDatRdd = rawDat.rdd.map(row => (row.getAs[String](0), row.getAs[String](1), row.getAs[Integer](2))).map(x => (x._1, x._2.split("\n").reverse, x._3))

    // attempt to remove some formatting
    val initCleanedRdd = rawDatRdd.map(x => {
      val allLines = x._2
      val cleanedLines = allLines.map(l => tryCleanLine(l, cleanCount=0)).filter(l => l.trim().nonEmpty) // remove lines that are empty when trimmed
      (x._1, cleanedLines, x._3)
    }).filter(_._2.length > 100) // filter for docs that still have more than 100 lines left; 193,170

    // first attempt to remove index-like stuff, conservatively dropping via dropWhile
    val stripIndex = initCleanedRdd.map(x => {
      val flippedDoc = x._2.map(l => (l.trim(), l)) // trimmed line, orig line
      // remove lines that start with numbers (until it reaches a line that does not)
      val removedIndex = flippedDoc.dropWhile(t => t._1(0).isDigit)
      // remove lines that satisfy the below special chars logic (halts when next line fails check)
      val droppedSpecialChars = removedIndex.dropWhile(t => {
        val l = t._1
        val numCommas = l.count(_ == ',')
        val numPipes = l.count(_ == '|')
        val numPeriods = l.count(_ == '.')
        val numUnderscores = l.count(_ == '_')
        val numDashes = l.count(_ == '-')
        val numHashes = l.count(_ == '#')
        val wordList = l.split(" ").map(_.trim()).filter(_.nonEmpty)
        val numWords = wordList.length
        val lastWord = wordList.last
        val numDigitsInLastWord = lastWord.count(_.isDigit)

        numDigitsInLastWord * 2 > lastWord.length || numCommas > numWords || (5*numPipes + 10*numPeriods + 3*numUnderscores + 3*numDashes + 3*numHashes) > l.length
      }).map(_._2)
      (x._1, droppedSpecialChars, x._3)
    }).filter(_._2.length > 100).repartition(outputParallelism) // filter for docs that still have more than 100 lines left, keep in reverse; 193,087

    // try stripping the footer more aggressively

    // Debug info: tally of (loopCounts, numbooks) after this line:
    //
    //  Array((1,106065), (2,14981), (3,33285), (4,12297), (5,8136), (6,6060), (7,4281), (8,3025), (9,1944), (10,1239), (11,775), (12,426),
    //  (13,220), (14,97), (15,62), (16,26), (17,26), (18,11), (19,6), (20,4), (21,126))

    val tryStripFooterRdd = stripIndex.map(x => {
      val emptyMatchList = mutable.Map[String, Int]().withDefaultValue(0)
      try {
        val (loopCount, fileName, strippedDoc, matchedList) = tryStripFooter(x._1, x._2, 0.1, 0,  emptyMatchList)
        (fileName, strippedDoc, x._3, loopCount, matchedList)
      }
      catch { // failed stripping for whatever reason
        case _: Throwable => (x._1, x._2, x._3, -1, emptyMatchList)
      }
    })
    tryStripFooterRdd.map(x => (x._4, 1)).reduceByKey(_+_).collect()

    val outputRdd = tryStripFooterRdd.filter(_._4 <= 10).map(x => Row(x._1, x._2, x._3, x._4))

    val schema = new StructType()
      .add(StructField("filename", StringType, nullable = false))
      .add(StructField("text", StringType, nullable = false))
      .add(StructField("header_strip_count", IntegerType, nullable = false))
      .add(StructField("footer_strip_count", IntegerType, nullable = false))

    val outputDf = spark.createDataFrame(outputRdd, schema)

    outputDf.repartition(240).write.mode(SaveMode.Overwrite)
      .parquet(s"${hdfsDir}/processed_parquet/${processDate}/stage2_strip_footer/")
    outputDf.repartition(240).write.mode(SaveMode.Overwrite)
      .json(s"${hdfsDir}/processed_json/${processDate}/stage2_strip_footer/")

  }
}
