@file:Suppress("PropertyName", "SpellCheckingInspection")

val tensorflowVersion: String by project

plugins {
  kotlin("jvm") version "1.4.21"
}

group = "cl.ravenhill"
version = "0.0.4"

repositories {
  jcenter()
  mavenCentral()
}

dependencies {
  implementation(kotlin("stdlib"))
  implementation(
    group = "org.tensorflow",
    name = "tensorflow-core-platform",
    version = tensorflowVersion
  )
  implementation("org.junit.jupiter:junit-jupiter:5.4.2")
  testImplementation("org.junit.jupiter:junit-jupiter:5.4.2")
}

tasks.test {
  useJUnitPlatform()
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
  kotlinOptions.jvmTarget = "13"
}