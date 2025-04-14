// Arduino Sketch for Speed Control via Serial (Receives 0-255 integer)

// Pin definitions for the motor driver (e.g., L298N H-Bridge)
const byte enablePin = 9; // PWM pin for speed control (must be a PWM pin, e.g., ~9 on Uno)
const byte in1Pin = 7;    // Control pin 1 for direction
const byte in2Pin = 8;    // Control pin 2 for direction

int currentPwmValue = 0; // Variable to store the current PWM speed value (0-255)

void setup() {
  Serial.begin(9600); // Match Python's SPEED_CONTROL_BAUD_RATE (9600)
  
  pinMode(enablePin, OUTPUT);
  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);

  // --- Set FIXED Motor Direction ---
  // Adjust HIGH/LOW combination based on your motor wiring
  // to achieve the desired default direction (e.g., forward).
  // This sketch assumes direction is constant and only speed changes via Serial.
  digitalWrite(in1Pin, HIGH); // Example: Set motor to move "forward"
  digitalWrite(in2Pin, LOW); 
  // -------------------------------

  // Stop motor initially
  analogWrite(enablePin, 0);
  currentPwmValue = 0;

  Serial.println("Speed Control Arduino Ready.");
  Serial.println("Fixed Direction: IN1=HIGH, IN2=LOW"); // Optional: Confirm fixed direction
  Serial.println("Waiting for PWM value (0-255) from Serial...");
}
void loop() {
  // Check if data is available to read from the serial port
  if (Serial.available() > 0) {
      
    // Read the incoming integer value. parseInt() stops at the first non-digit.
    int receivedValue = Serial.parseInt(); 

    // Consume the rest of the line, including the newline character ('\n')
    // sent by the Python script, to keep the buffer clean.
    while (Serial.available() > 0 && Serial.peek() != '\n') { 
        Serial.read(); // Read and discard characters until newline or empty
    }
    if (Serial.available() > 0 && Serial.peek() == '\n') {
        Serial.read(); // Read and discard the newline itself
    }

    // Constrain the received value to the valid PWM range (0-255)
    int newPwmValue = constrain(receivedValue, 0, 255);

    // Only update the motor speed if the constrained value is different
    // from the current speed. This avoids unnecessary analogWrite calls.
    if (newPwmValue != currentPwmValue) {
      currentPwmValue = newPwmValue; // Store the new valid value
      analogWrite(enablePin, currentPwmValue); // Set the PWM speed on the enable pin

      // Print confirmation to Serial Monitor for debugging
      Serial.print("PWM speed set to: ");
      Serial.println(currentPwmValue); 
    }
  }
  
  // The loop continues, maintaining the last set PWM speed.
  // The fixed direction set in setup() remains active.
}