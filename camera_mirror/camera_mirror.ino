#include <Servo.h>

Servo mirrorLR; // left/right rotation
Servo mirrorUD; // up/down rotation
Servo camera;

// set 0 positions for servos
int mirrorLRHome = 35;
int mirrorUDHome = 55;

int mirrorLRAngle = mirrorLRHome; 
int mirrorUDAngle = mirrorUDHome;

int cameraHome = 95;
int cameraAngle = cameraHome;

void setup() {
  mirrorLR.attach(9); 
  mirrorUD.attach(10); 
  camera.attach(7); 
  
  // start at home position
  mirrorLR.write(mirrorLRAngle);
  mirrorUD.write(mirrorUDAngle);
  camera.write(cameraAngle);

  Serial.begin(9600);  
}

void loop() {
  if (Serial.available() == 2) {
    char device = Serial.read();
    char input = Serial.read();

    if (device == 'C') {
        switch(input) {
          case 'R': 
            cameraAngle = max(cameraAngle - 1, 0); 
            camera.write(cameraAngle);
            break;
            
          case 'L':
            cameraAngle = min(cameraAngle + 1, 180);  
            camera.write(cameraAngle);
            break;
          
          case 'H':
            camera.write(cameraHome);
            break;

          default:
            break;
        }
    } else if (device == 'M') {
        switch (input) {
          case 'R':
            mirrorLRAngle = max(mirrorLRAngle - 1, 0); 
            mirrorLR.write(mirrorLRAngle);
            break;
            
          case 'L':
            mirrorLRAngle = min(mirrorLRAngle + 1, 180);  
            mirrorLR.write(mirrorLRAngle);
            break;

          case 'U':
            mirrorUDAngle = max(mirrorUDAngle - 1, 0); 
            mirrorUD.write(mirrorUDAngle);
            break;
            
          case 'D':
            mirrorUDAngle = min(mirrorUDAngle + 1, 70); 
            mirrorUD.write(mirrorUDAngle);
            break;

          case 'H':
            mirrorLR.write(mirrorLRHome);
            mirrorUD.write(mirrorUDHome);
            break;

          default:
            break;
      }
  }
}
}
