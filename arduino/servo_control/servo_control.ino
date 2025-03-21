#include <Servo.h>

int y_pos = 0;
int y_pos_min = 0;
int y_pos_max = 90;
int y_speed = 1;
Servo myservo;

int x_pos = 0;

int x_error, y_error;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  myservo.attach(9); // attaches the servo on pin 9
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  y_pos = 0;
  myservo.write(y_pos);
}

void loop() {
  // put your main code here, to run repeatedly:

  if (Serial.available() > 0) {
        digitalWrite(LED_BUILTIN, HIGH);
        String data = Serial.readStringUntil('\n');
        // Serial.println("Received: " + received);  // Print to Serial Monitor
        data.trim();
        // Find the positions of the delimiters
        int x_error_start = data.indexOf("X_error:") + 8; // Start of X_error value
        int comma_pos = data.indexOf(','); // Position of the comma
        int y_error_start = data.indexOf("Y_error:") + 8; // Start of Y_error value

        // Extract X_error and Y_error as substrings
        String x_error_str = data.substring(x_error_start, comma_pos);
        String y_error_str = data.substring(y_error_start);

        // Check if the datas are not blank
        if ((x_error_str.length() > 0) && (y_error_str.length() > 0)) {
          // Convert the strings to floats
          x_error = x_error_str.toInt();
          y_error = y_error_str.toInt();  

          if ((y_error > 100) && (y_pos < y_pos_max)) {
              y_pos += y_speed;
              myservo.write(y_pos);
              delay(15); 
          }
          else if ((y_error < -100) && (y_pos > y_pos_min)) {
              y_pos -= y_speed;
              myservo.write(y_pos);
              delay(15);         
          }
        } 
  }
  else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}


