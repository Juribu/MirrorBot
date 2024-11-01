#include <Servo.h>

Servo servo1; // left/right rotation
Servo servo2; // up/down rotation

// set 0 positions for servos
int servo1Home = 25;
int servo2Home = 55;

int angle1 = servo1Home; 
int angle2 = servo2Home;

void setup() {
  servo1.attach(9); 
  servo2.attach(10); 
  
  // start at home position
  servo1.write(angle1);
  servo2.write(angle2);

  Serial.begin(9600);  
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();  

    switch (input) {
      case 'R':
        angle1 = max(angle1 - 15, 0); 
        servo1.write(angle1);
        break;
        
      case 'L':
        angle1 = min(angle1 + 15, 180);  
        servo1.write(angle1);
        break;

      case 'U':
        angle2 = max(angle2 - 15, 0); 
        servo2.write(angle2);
        break;
        
      case 'D':
        angle2 = min(angle2 + 15, 180); 
        servo2.write(angle2);
        break;

      case 'H':
        servo1.write(servo1Home);
        servo2.write(servo2Home);
        break;

      default:
       
        break;
    }
  }
}

