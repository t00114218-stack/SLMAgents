# Keep JNI native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep Room database entities
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-keepclassmembers class * {
    @androidx.room.PrimaryKey *;
    @androidx.room.ColumnInfo *;
}
