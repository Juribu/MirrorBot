#include <Servo.h>

Servo mirrorX; // left/right rotation
Servo mirrorY; // up/down rotation
Servo camera;

// set 0 positions for servos
int cameraHome = 95;
int mirrorXHome = 34;
int mirrorYHome = 55;

int mirrorXAngle = mirrorXHome; 
int mirrorYAngle = mirrorYHome;
int cameraAngle = cameraHome;

void setup() {
  mirrorX.attach(9); 
  mirrorY.attach(10); 
  camera.attach(7); 
  
  // start at home position
  mirrorX.write(mirrorXAngle);
  mirrorY.write(mirrorYAngle);
  camera.write(cameraAngle);

  Serial.begin(9600);  
}

void loop() {
  if (Serial.available() > 0) {
    // input format: <device> <angle>\n
    // angle is from -90 to 90 degrees
    String device = Serial.readStringUntil(' ');
    int angle = Serial.readStringUntil('\n').toInt();
    Serial.println("Received: " + angle);
    // angle = constrain(angle, -90, 90);

    if (device == "C") {
      cameraAngle = cameraHome + angle;
      camera.write(cameraAngle);
    } else if (device == "MX") {
      mirrorXAngle = mirrorXHome + angle;
      mirrorX.write(mirrorXAngle);
    } else if (device == "MY") {
      mirrorYAngle = mirrorYHome + angle;
      mirrorY.write(mirrorYAngle);
    }  
  }
}
