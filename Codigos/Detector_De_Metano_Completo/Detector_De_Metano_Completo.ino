#include <WiFi.h>
#include <WiFiUdp.h>
#include "max6675.h"
#include <Wire.h>


#define LED0 2
#define LED1 15
#define RXD2 16
#define TXD2 17
#define CONFIG_TCSCK_PIN      18
#define CONFIG_TCCS_PIN       13
#define CONFIG_TCDO_PIN       19

#define PUMP 5
#define Flow_Pin 34
#define Methane_Pin 35

double Flow = 0.0;
double Methane = 0.0;
double Kp = 0.0;
double Kd = 0.0;
double Ki = 0.0;
double Set_flow = 500;

const char* ssid = "WeWork";
const char* password = "P@ssw0rd";

MAX6675 thermocouple(CONFIG_TCSCK_PIN, CONFIG_TCCS_PIN, CONFIG_TCDO_PIN);

int Id_client = 255; //Identificación del cliente
int p = 0; //Indicador de cada dato (paquete) enviado
int contconexion = 0;
double Temp = 0.0;
WiFiUDP     Udp;
#include <MPUandes.h>

const int scl = 22;
const int sda = 21;

int AccelScaleFactor = 2048;
int GyroScaleFactor = 131;

MPUandes acelerometro = MPUandes(sda, scl);

void setup()
{
  pinMode(LED0, OUTPUT);
  pinMode(LED1, OUTPUT);
  digitalWrite(LED0, 1);
  Serial.begin(115200);
  Wire.begin(sda, scl);
  delay(100);
  acelerometro.MPU6050_Init();
  delay(2000);
  // Setting The Mode Of Pins

  pinMode(Flow_Pin, INPUT);
  pinMode(Methane_Pin, INPUT);
  pinMode(PUMP, OUTPUT);

  WiFi.mode(WIFI_STA); //para que no inicie el SoftAP en el modo normal
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED and contconexion < 50) //Cuenta hasta 50 si no se puede conectar lo cancela
  {
    ++contconexion;
    delay(250);
    Serial.print(".");
  }
  if (contconexion < 50)
  {
    //Se está usando el código con IP dinámica por problemas con la ip fija. (funciona igualmente bien con IP dinámica)
    Serial.println("");
    Serial.println("WiFi conectado");
    Serial.println(WiFi.localIP());
    Serial.println(WiFi.gatewayIP());
  }
  else
  {
    Serial.println("");
    Serial.println("Error de conexion");
  }
  Serial2.begin(38400, SERIAL_8N2, RXD2, TXD2);
  delay(1000);
  Serial2.print("[A]");
  Serial.println("Modo normal activado");
  delay(1000);
  digitalWrite(PUMP, 1);
  digitalWrite(LED0, 0);
}
int Ax1, Ay1, Az1, Gx1, Gy1, Gz1;
double Ax, Ay, Az, Gx, Gy, Gz;
String msg = "";
String ppm;
String ppm_CH4 = "00000000";
boolean takeNext = false;
long t = 0.0;
boolean flag = true;
long number = 0;
void loop()
{
  Flow = analogRead(Flow_Pin);
  Methane = analogRead(Methane_Pin);

  while (Serial2.available() >= 0 && flag == true)
  {
    ppm = Serial2.readStringUntil('\n');
    if (takeNext)
    {
      ppm_CH4 = ppm;
      takeNext = false;
      number = (int) strtol(&ppm_CH4[1], NULL, 16);
      flag = false;
      Serial.print(Methane);
      Serial.print("  ,   ");
      Serial.println(number);
      
    }
    if (ppm == "0000005b")
    {
      takeNext = true;
    }
  }


  Ax1 = (int)acelerometro.Read_Ax();
  Ay1 = (int)acelerometro.Read_Ay();
  Az1 = (int)acelerometro.Read_Az();
  Gx1 = (int)acelerometro.Read_Gx();
  Gy1 = (int)acelerometro.Read_Gy();
  Gz1 = (int)acelerometro.Read_Gz();
  //divide each with their sensitivity scale factor
  Ax = (double)Ax1 / AccelScaleFactor;
  Ay = (double)Ay1 / AccelScaleFactor;
  Az = (double)Az1 / AccelScaleFactor;
  Gx = (double)Gx1 / GyroScaleFactor;
  Gy = (double)Gy1 / GyroScaleFactor;
  Gz = (double)Gz1 / GyroScaleFactor;
  if ( flag == false)
  {
    digitalWrite(LED0, 1);
    Udp.beginPacket("10.172.0.57", 9001);  //// Esta ip es la ip del computador servidor y el puerto debe coincidir

    //digitalWrite(LED1, HIGH);
    msg = String(Ax) + "#" + String(Ay) + "#" + String(Az) + "#" + String(Gx) + "#" + String(Gy) + "#" + String(Gz) + "#" + String(Id_client) + "#" + String(p) + "#" + String(number); //El mensaje completo contiene el id del cliente y el numero de paquete enviado

    for (int i = 0; i < msg.length(); i++)
    {
      int old = micros();
      Udp.write(msg[i]);
      while (micros() - old < 87);
    }

    Udp.endPacket();
    t = millis();
    p = p + 1;
    Serial.println(msg);
    flag = true;
    digitalWrite(LED0, 0);
  }
}
