#define RXD2 16
#define TXD2 17
#define PUMP 5
#define LED_0 2

bool takeNext = false;
long analogo = 0;
long medido = 0;
void setup()
{
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(LED_0, OUTPUT);
  pinMode(PUMP, OUTPUT);

  Serial2.begin(38400, SERIAL_8N2, RXD2, TXD2);
  delay(1000);
  Serial2.println("[A]");
  delay(1000);
  
}
int number = 0;
boolean state = 0;
void loop()
{
  analogo = analogRead(35);
  if (Serial.available() >= 0)
  {
    char y = Serial.read();
    if (y == '1')
    {
      Serial2.println("[A]");
      Serial.println("nomal mode");
    }
    if (y == '2')
    {
      Serial2.println("[C]");
      Serial.println("configuration mode");
    }
    if (y == '3')
    {
      Serial2.println("[I]");
      Serial.println("settings");
    }
    if (y == '4')
    {
      Serial2.println("[K]");
      Serial.println("reset");
    }
    if (y == '5')
    {
      Serial2.println("[E]");
      Serial.println("Zero calibrate");
    }
    if (y == '6')
    {
      Serial2.println("[B]");
      Serial.println("ENGINERING Mode");
    }
    if (y == '7')
    {
      state =! state;
      digitalWrite(PUMP, state);
      Serial.println(state);
    }
     if (y == '8')
    {
      Serial2.println("[N0000000300001388]");
      Serial.println("load span");
    }
    
  }
  if (Serial2.available())
  {
    String x = Serial2.readStringUntil('\n');
    //    if (takeNext)
    //    {
    //      medido = (int) strtol(&x[1], NULL, 16);
    //      takeNext = false;
    //    }
    //    if (x == "0000005b")
    //    {
    //      takeNext = true;
    //    }
    if (x == "0000005b")
    {
      Serial.println("Comienzo de paquete");
    }
    else if (x == "0000005d")
    {
      Serial.println("final de paquete");
    }
    else if (x == "5b414b5d")
    {
      Serial.println("[AK]");
    }
    else if (x == "5b4e415d")
    {
      Serial.println("[NA]");
    }
    else
    {
      number = (int) strtol(&x[1], NULL, 16);
      Serial.print(x);
      Serial. print("   ,   ");
      Serial.println(number);
    }

  }
  //  Serial.print("Serial   :");
  //  Serial.print(medido);
  //  Serial.print("  ,  Analogo   :");
  //  Serial.println(analogo);

}
