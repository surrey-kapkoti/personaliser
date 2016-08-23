name := "RecommenderSystem"

version := "0.0.1"

scalaVersion := "2.11.8"

// additional libraries
//javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

// protocol buffer support
//seq(sbtprotobuf.ProtobufPlugin.protobufSettings: _*)

// additional libraries
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.0.0",
  "org.apache.spark" %% "spark-hive" % "2.0.0",
  "it.nerdammer.bigdata" % "spark-hbase-connector_2.10" % "1.0.3",
  "org.apache.spark" %% "spark-mllib" % "2.0.0"
)

resolvers ++= Seq(
  "JBoss Repository" at "http://repository.jboss.org/nexus/content/repositories/releases/",
  "Spray Repository" at "http://repo.spray.cc/",
  "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/",
  "Akka Repository" at "http://repo.akka.io/releases/",
  //"Twitter4J Repository" at "http://twitter4j.org/maven2/",
  //"Apache HBase" at "https://repository.apache.org/content/repositories/releases",
  //"Twitter Maven Repo" at "http://maven.twttr.com/",
  "scala-tools" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
  "Mesosphere Public Repository" at "http://downloads.mesosphere.io/maven",
  Resolver.sonatypeRepo("public")
)

