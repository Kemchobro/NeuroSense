plugins {
	kotlin("jvm") version "1.9.24"
	application
}

repositories {
	mavenCentral()
}

dependencies {
	implementation(kotlin("stdlib"))
	implementation("com.github.wendykierp:JTransforms:3.1")
}

// Configure Java compatibility - Java 23 is backward compatible with Java 17
java {
	sourceCompatibility = JavaVersion.VERSION_17
	targetCompatibility = JavaVersion.VERSION_17
}

// Configure Kotlin to target Java 17 bytecode (compatible with Java 23 runtime)
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
	kotlinOptions.jvmTarget = "17"
}

application {
	mainClass.set("GenerateSpectrogramsKt")
}
