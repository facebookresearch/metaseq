// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import ProjectDependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.meta",
      scalaVersion := "2.12.10",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "BooksCleaner",
    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.2",
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.1.2",
    libraryDependencies += scalaTest % Test
  )

// Source: https://stackoverflow.com/questions/31570503/get-reference-to-jar-assembly-path-in-sbt/31592840
//  Will use this to print the artifact we need to copy over to the cluster
lazy val artifactPathExt = settingKey[String]("Get the main artifact path")
artifactPathExt.withRank(KeyRanks.Invisible) := (artifactPath in (Compile, packageBin)).value.getPath

console / initialCommands := """
  import org.apache.spark.sql.SparkSession
  import org.apache.spark.sql.functions._
  val spark = SparkSession.builder
    .master("local[*]")
    .appName("spark-shell")
    .getOrCreate
"""

console / cleanupCommands := "if (spark != null) spark.stop()"
