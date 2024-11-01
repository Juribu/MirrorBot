#include <Servo.h>

Servo servo;

int servoHome = 90;
int angle = servoHome;


void setup() {
  servo.attach(7); 
  servo.write(angle);
  Serial.begin(9600);  
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();  

    switch (input) {
      case 'R': 
        angle = max(angle - 2, 0); 
        servo.write(angle);
        break;
        
      case 'L':
        angle = min(angle + 2, 180);  
        servo.write(angle);
        break;
      
      case 'H':
        servo.write(servoHome);
        break;

      default:
       
        break;
    }
  }
}

