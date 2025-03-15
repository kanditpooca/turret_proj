#include <Servo.h>

Servo myServo; // Create a servo object

int targetPosition = 90; // Target position of the servo
int currentPosition = 0;  // Current position of the servo
const int speed = 5;      // Speed of movement (degrees per loop iteration)
const int delayTime = 50; // Delay between steps (controls overall speed)

void setup() {
  Serial.begin(9600);
  myServo.attach(9); // Attach the servo to pin 9
  myServo.write(currentPosition); // Initialize servo position
  delay(2000);
}

void loop() {
  // Move the servo gradually toward the target position
  if (currentPosition < targetPosition) {
    currentPosition += speed; // Move up
    if (currentPosition > targetPosition) {
      currentPosition = targetPosition; // Prevent overshooting
    }
  } else if (currentPosition > targetPosition) {
    currentPosition -= speed; // Move down
    if (currentPosition < targetPosition) {
      currentPosition = targetPosition; // Prevent overshooting
    }
  }

  // Set the servo position
  myServo.write(currentPosition);

  // Debugging output
  Serial.print("Current Position: ");
  Serial.println(currentPosition);

  delay(delayTime); // Control the speed of movement
}