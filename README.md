# Vehicle Inspection App

## Description
The Vehicle Inspection App leverages artificial intelligence to automatically detect vehicle damages and dents, streamlining the inspection process. This app simplifies vehicle assessments, providing accurate results and improving inspection efficiency.

## Setup Instructions
1. **Clean the project**: Run the following command to clean the project and remove any cached data:
   ```bash
   flutter clean
   flutter pub get


## setup for Android
1. Add this in the android/app/build.gradle file under the appropriate section.

```java
buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig signingConfigs.debug
        }
    }
```
