plugins {
  kotlin("jvm") version "1.4.21"
}

group = "cl.ravenhill"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  jcenter()
  maven(url = "https://kotlin.bintray.com/kotlin-datascience")
}

dependencies {
  implementation(kotlin("stdlib"))
  implementation(group="org.tensorflow", name="tensorflow-core-platform", version="0.2.0")
  implementation("org.jetbrains.kotlin-deeplearning:api:0.1.0")
}
