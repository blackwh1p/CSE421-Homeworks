#include "mbed.h"

#define WAIT_TIME_MS 200

AnalogIn intTemp(ADC_TEMP);

int main()
{
    const float VREF = 3.3f;     // VDDA (approximately)
    const float V25 = 0.76f;    // VSENSE (V) at 25°C
    const float AVG_SLOPE = 0.0025f;  // V/°C  (2.5 mV/°C)

    for (int i = 0; i < 10; i++)
    {
        uint16_t raw = intTemp.read_u16();               // 0-65535
        float vsense = (raw / 65535.0f) * VREF;          // Volt
        float tempC = ((vsense - V25) / AVG_SLOPE) + 25.0f;

        printf("[%02d] raw = %5u  Temp = %.2f C\r\n", i + 1, (unsigned)raw, tempC);
        thread_sleep_for(WAIT_TIME_MS);
    }

    printf("Done.\r\n");
    while (true)
    {
        thread_sleep_for(1000);
    }
}
