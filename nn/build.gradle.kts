plugins {
    alias(libs.plugins.android.library)
    id("com.chaquo.python")
}

android {
    namespace = "com.example.nn"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters.addAll(arrayOf("arm64-v8a"))
            abiFilters.addAll(libs.versions.ndkAbiFilters.get().split(','))
        }


    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild {
        cmake {
            version="3.10.2.4988404"
            path=File("CMakeLists.txt")
        }
    }
    ndkVersion="25.1.8937393"
}
chaquopy {
    defaultConfig {
        buildPython("C:\\Users\\30585\\AppData\\Local\\Programs\\Python\\Python311\\python.exe")
        pip {
            install("numpy")
            install("Jinja2")
            install("scipy")
            install("tokenizers")
        }
    }
}
dependencies {


    androidTestImplementation(libs.onnx.runtime)
}