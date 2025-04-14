#include <TinyGPS++.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050.h>

// GPS Configuration
static const int RXPin = 4, TXPin = 3;
SoftwareSerial ss(RXPin, TXPin);
TinyGPSPlus gps;

void setup() {
  Serial.begin(115200);
  ss.begin(9600);
  Wire.begin();
}

void loop() {
  // Process GPS data
  bool newGPSData = false;
  while (ss.available() > 0) {
    if (gps.encode(ss.read())) {
      newGPSData = true;
    }
  }

  // Print only GPS data when available
  if (newGPSData && gps.location.isValid()) {
    Serial.print("LAT=");
    Serial.print(gps.location.lat(), 6);
    Serial.print(",LON=");
    Serial.println(gps.location.lng(), 6);
  }
}
