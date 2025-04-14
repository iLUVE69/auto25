#include <Arduino.h> // Include base Arduino functions explicitly

// --- Encoder Pins and Variables ---
const byte encoder0PinA = 2; // Use byte for pin numbers
const byte encoder0PinB = 3;
volatile int encoder0Pos = 0;        // Current encoder position
volatile byte encoder0PinALast = LOW; // Use byte for single-bit state
volatile byte m = LOW;               // Use byte for pin B state

// --- Motor Control Pins ---
const byte pwmPin = 5;     // PWM output pin (e.g., ~5 on Uno)
const byte dirPin = 4;     // Direction control pin

// --- Control Variables ---
const int SET_MOVEMENT_PWM = 220; // Fixed PWM value used ONLY when moving to a SET target
long targetPosition = 0;         // Target encoder position for SET command
int currentPwmValue = 0;         // Stores the last value received from PWM: command (-255 to 255)

// --- State Machine for Motor Control ---
enum MotorMode {
  MODE_PWM_CONTROL, // Following external PWM commands directly
  MODE_SET_TARGET   // Actively moving towards a target position set by SET: command
};
MotorMode currentMode = MODE_PWM_CONTROL; // Start in PWM mode (effectively stopped until command received)

// --- Serial Communication ---
String inputString = "";      // A String to hold incoming data
boolean stringComplete = false; // Whether the string is complete

// --- Setup ---
void setup() {
  pinMode(encoder0PinA, INPUT_PULLUP);
  pinMode(encoder0PinB, INPUT_PULLUP);
  pinMode(pwmPin, OUTPUT);
  pinMode(dirPin, OUTPUT);

  Serial.begin(9600); // Match Python's MOTOR_CONTROL_BAUD_RATE
  inputString.reserve(50); // Pre-allocate string space for efficiency

  // Initialize encoder reading safely
  cli(); // Disable interrupts temporarily
  encoder0PinALast = digitalRead(encoder0PinA);
  m = digitalRead(encoder0PinB);
  encoder0Pos = 0; // Start position at 0
  sei(); // Enable interrupts

  // Attach interrupts AFTER reading initial state and setting position
  attachInterrupt(digitalPinToInterrupt(encoder0PinA), CountA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoder0PinB), StateB, CHANGE);

  // Ensure motor is stopped initially
  stopMotor();
  currentMode = MODE_PWM_CONTROL; // Default state

  Serial.println("Arduino Ready. Waiting for SET: or PWM: commands...");
}

// --- Main Loop ---
void loop() {
  // Atomically read the volatile encoder position
  int localEncoderPos;
  noInterrupts(); // Start critical section - disable interrupts
  localEncoderPos = encoder0Pos;
  interrupts();   // End critical section - enable interrupts

  // --- Handle Serial Input ---
  serialEvent(); // Check for and buffer incoming serial data

  if (stringComplete) {
    inputString.trim(); // Remove leading/trailing whitespace
    // Serial.print("Received: "); Serial.println(inputString); // Debug: Echo command

    // --- Command Parsing ---
    if (inputString.startsWith("PWM:")) {
      // Parse and handle the PWM command immediately
      parseAndHandlePwmCommand(inputString);

    } else if (inputString.startsWith("SET:")) {
      // Parse the SET command and change state
      parseAndHandleSetCommand(inputString);

    } else if (inputString.length() > 0) { // Report unknown commands (ignore empty lines)
      Serial.print("Unknown command: "); Serial.println(inputString);
    }

    // Clear the string ready for the next command
    inputString = "";
    stringComplete = false;
  }

  // --- State-Based Motor Control ---
  // This block ONLY runs the logic for moving towards a SET target
  if (currentMode == MODE_SET_TARGET) {
    // Compare current position with the target
    if (localEncoderPos < targetPosition) {
      // Need to move forward (Adjust LOW/HIGH based on your specific motor wiring)
      digitalWrite(dirPin, LOW); // Example: LOW = forward
      analogWrite(pwmPin, SET_MOVEMENT_PWM); // Use the fixed PWM for SET movement
    } else if (localEncoderPos > targetPosition) {
      // Need to move backward (Adjust LOW/HIGH based on your specific motor wiring)
      digitalWrite(dirPin, HIGH); // Example: HIGH = backward
      analogWrite(pwmPin, SET_MOVEMENT_PWM); // Use the fixed PWM for SET movement
    } else {
      // Target reached!
      stopMotor();
      currentMode = MODE_PWM_CONTROL; // Switch back to PWM control mode (which is now stopped)
      currentPwmValue = 0;          // Reset the stored PWM value
      Serial.println("Target reached!"); // Acknowledge Python task completion
    }
  }
  // NOTE: If currentMode == MODE_PWM_CONTROL, the motor pins were already set
  // when the "PWM:" command was parsed in parseAndHandlePwmCommand().
  // Nothing further needs to be done in the loop for that mode.

  // --- Optional: Print encoder position ---
  static int valOld = 0; // Keep track of the last printed value
  if (localEncoderPos != valOld) {
    // Serial.print("Encoder Position: "); // Uncomment for frequent debugging
    // Serial.println(localEncoderPos);    // Uncomment for frequent debugging
    valOld = localEncoderPos;
  }
  // delay(5); // Small delay if printing frequently
}

// --- Interrupt Service Routines (ISRs) ---
// Function called when pin A changes state
void CountA() {
  // Read pin A and B states
  byte n_state = digitalRead(encoder0PinA);
  // Note: Reading 'm' directly here relies on it being updated quickly enough by StateB ISR.
  // If high speeds cause issues, 'm' could be read inside this ISR too.
  if ((encoder0PinALast == LOW) && (n_state == HIGH)) { // Rising edge on A
    // Check pin B state to determine direction
    encoder0Pos += (m == LOW) ? -1 : 1; // CW or CCW (adjust + / - if reversed)
  }
  encoder0PinALast = n_state; // Store last state of A
}

// Function called when pin B changes state
void StateB() {
  m = digitalRead(encoder0PinB); // Update the global state of pin B
}

// --- Serial Event Function (Called automatically when serial data arrives) ---
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    // If the incoming character is a newline, set a flag so the main loop can
    // process the string. This prevents blocking inside serialEvent.
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}

// --- Command Handling Functions ---

// Parses "PWM: <value>" and directly controls the motor
void parseAndHandlePwmCommand(String cmd) {
  // Extract value part after "PWM:" (index 4)
  String valueStr = cmd.substring(4);
  valueStr.trim();
  int value = valueStr.toInt(); // Convert to integer (-255 to 255 expected)

  // Serial.print("Parsed PWM value: "); Serial.println(value); // Debug

  // --- Apply Control ---
  // Switch mode FIRST, so any SET logic stops
  currentMode = MODE_PWM_CONTROL;
  // Store and clamp the value
  currentPwmValue = constrain(value, -255, 255);

  if (currentPwmValue == 0) {
    stopMotor();
  } else if (currentPwmValue > 0) {
    // Forward / Right (Example - adjust LOW/HIGH for your wiring)
    digitalWrite(dirPin, LOW);
    analogWrite(pwmPin, currentPwmValue); // analogWrite handles 0-255
  } else { // value < 0
    // Backward / Left (Example - adjust LOW/HIGH for your wiring)
    digitalWrite(dirPin, HIGH);
    analogWrite(pwmPin, abs(currentPwmValue)); // Use absolute value for PWM magnitude
  }
}

// Parses "SET: <position>" and prepares for SET target movement
void parseAndHandleSetCommand(String cmd) {
  // Extract value part after "SET:" (index 4)
  String valueStr = cmd.substring(4);
  valueStr.trim();
  // Use .toInt() or .toLong() depending on expected range of positions
  targetPosition = valueStr.toInt();

  Serial.print("New target position set: "); Serial.println(targetPosition);

  // --- Change State ---
  // Stop any previous PWM control explicitly before starting SET movement logic
  stopMotor();
  currentPwmValue = 0; // Reset stored PWM value
  // Set the mode so the main loop starts the SET movement logic
  currentMode = MODE_SET_TARGET;

  // Optional: Acknowledge command received if Python needs it
  // Serial.println("ACK SET Received");
}

// --- Utility Function ---
// Stops the motor by setting PWM to 0
void stopMotor() {
  analogWrite(pwmPin, 0);
  // Consider setting dirPin to a default state (e.g., LOW) if your driver requires it
  // digitalWrite(dirPin, LOW);
  // Serial.println("Motor Stopped"); // Debug
}