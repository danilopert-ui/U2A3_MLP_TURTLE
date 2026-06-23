/*
 * ============================================================
 *  U2A3 - Perceptron Multicapa (MLP) para robot TURTLE
 *  Seguidor de linea: 3 sensores de entrada -> 2 motores (diferencial)
 * ------------------------------------------------------------
 *  Arquitectura : 3 - 4 - 2
 *    - Entrada   : 3 neuronas (sensor izquierdo, central, derecho)
 *    - Oculta    : 4 neuronas, activacion tanh
 *    - Salida    : 2 neuronas, activacion sigmoide (0..1) -> PWM motores
 *
 *  El Arduino SOLO ejecuta la propagacion hacia adelante (forward pass).
 *  Los pesos y bias fueron entrenados FUERA del microcontrolador
 *  (Python, descenso de gradiente, seed=0) y se cargan como constantes.
 *
 *  Plataforma: Arduino UNO (ATmega328P) en Proteus, robot TURTLE.
 *  Autor: Angel Daniel
 * ============================================================
 */

#include <math.h>

// ---------------- Asignacion de pines ----------------
// Ajusta estos pines para que coincidan con tu cableado en Proteus.
const uint8_t PIN_SENSOR_IZQ = A0;   // sensor de linea izquierdo
const uint8_t PIN_SENSOR_CEN = A1;   // sensor de linea central
const uint8_t PIN_SENSOR_DER = A2;   // sensor de linea derecho
const uint8_t PIN_MOTOR_IZQ  = 5;    // motor izquierdo (salida PWM)
const uint8_t PIN_MOTOR_DER  = 6;    // motor derecho  (salida PWM)

// ---------------- Topologia de la red ----------------
const uint8_t N_IN  = 3;
const uint8_t N_H   = 4;
const uint8_t N_OUT = 2;

// ---------------- Pesos y bias entrenados ----------------
// W1: pesos entrada -> oculta (N_H x N_IN)
const float W1[N_H][N_IN] = {
  {1.329676f, 2.018578f, 0.999286f},
  {1.590658f, 0.960025f, -1.111810f},
  {1.192569f, 0.104363f, 0.457913f},
  {-0.485088f, 0.825060f, 1.413771f}
};
// b1: bias de la capa oculta
const float b1[N_H] = { -0.349252f, 0.008270f, -0.210004f, -0.276794f };

// W2: pesos oculta -> salida (N_OUT x N_H)
const float W2[N_OUT][N_H] = {
  {1.199439f, -1.115519f, -1.162569f, 1.391347f},
  {1.762919f, 1.108431f, -0.270646f, -1.796346f}
};
// b2: bias de la capa de salida
const float b2[N_OUT] = { 0.548428f, 0.043954f };

// ---------------- Funciones de activacion ----------------
float activacionOculta(float z) {
  return tanhf(z);              // tanh en la capa oculta
}

float activacionSalida(float z) {
  return 1.0f / (1.0f + expf(-z));  // sigmoide en la salida (0..1)
}

// ---------------- Lectura de sensores ----------------
// Devuelve 1 si el sensor detecta la linea, 0 en caso contrario.
// Si la polaridad esta invertida en tu escenario, niega la lectura.
void leerSensores(float x[N_IN]) {
  x[0] = (float) digitalRead(PIN_SENSOR_IZQ);
  x[1] = (float) digitalRead(PIN_SENSOR_CEN);
  x[2] = (float) digitalRead(PIN_SENSOR_DER);
}

// ---------------- Propagacion hacia adelante (inferencia) ----------------
void mlpForward(const float x[N_IN], float y[N_OUT]) {
  float a1[N_H];

  // Capa oculta: z = W1*x + b1 ; a1 = tanh(z)
  for (uint8_t j = 0; j < N_H; j++) {
    float z = b1[j];
    for (uint8_t i = 0; i < N_IN; i++) {
      z += W1[j][i] * x[i];
    }
    a1[j] = activacionOculta(z);
  }

  // Capa de salida: z = W2*a1 + b2 ; y = sigmoide(z)
  for (uint8_t k = 0; k < N_OUT; k++) {
    float z = b2[k];
    for (uint8_t j = 0; j < N_H; j++) {
      z += W2[k][j] * a1[j];
    }
    y[k] = activacionSalida(z);
  }
}

// ---------------- Mapeo salida -> PWM del motor ----------------
uint8_t aPWM(float salida) {
  float v = salida * 255.0f;            // 0..1  ->  0..255
  return (uint8_t) constrain(v, 0, 255);
}

// ---------------- Configuracion ----------------
void setup() {
  pinMode(PIN_SENSOR_IZQ, INPUT);
  pinMode(PIN_SENSOR_CEN, INPUT);
  pinMode(PIN_SENSOR_DER, INPUT);
  pinMode(PIN_MOTOR_IZQ, OUTPUT);
  pinMode(PIN_MOTOR_DER, OUTPUT);
  Serial.begin(9600);
}

// ---------------- Bucle principal ----------------
void loop() {
  float x[N_IN];
  float y[N_OUT];

  leerSensores(x);          // 1) leer sensores
  mlpForward(x, y);         // 2) ejecutar el MLP (forward)

  // 3) interpretar salidas: y[0] -> motor izquierdo, y[1] -> motor derecho
  uint8_t pwmIzq = aPWM(y[0]);
  uint8_t pwmDer = aPWM(y[1]);

  // 4) controlar motores
  analogWrite(PIN_MOTOR_IZQ, pwmIzq);
  analogWrite(PIN_MOTOR_DER, pwmDer);

  // --- Depuracion por monitor serie ---
  Serial.print("Sensores ");
  Serial.print((int)x[0]); Serial.print((int)x[1]); Serial.print((int)x[2]);
  Serial.print("  ->  PWM_izq=");
  Serial.print(pwmIzq);
  Serial.print("  PWM_der=");
  Serial.println(pwmDer);

  delay(20);
}
