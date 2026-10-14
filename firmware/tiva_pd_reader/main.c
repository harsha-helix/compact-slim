//*****************************************************************************
// main.c - TM4C123GXL (Tiva C LaunchPad) photodiode ADC reader.
//
// Built on TivaWare's ek-tm4c123gxl "blinky" example (same CCS project,
// device, driverlib and linker script). Streams the photodiode reading over
// UART0 (the LaunchPad's ICDI virtual COM port, e.g. COM21 on Windows) in
// the frame format photonic_annealing.py's _pd_reader_worker expects:
//
//     [0xA5][val_lo][val_hi]      (3 bytes, 12-bit ADC value, little-endian,
//                                  val_hi always <= 0x0F)
//
// Host side converts with  voltage = (val / 4095.0) * 3.3
//
// --- Noise reduction ---
// Every reported sample is the average of ADC_HW_OVERSAMPLE raw conversions
// done in ADC hardware (zero CPU cost), and SW_ACCUM_DEPTH of those are
// further averaged in software before one frame is sent. So each frame is
// the average of ADC_HW_OVERSAMPLE * SW_ACCUM_DEPTH = 256 raw samples taken
// over the last 1/OUTPUT_RATE_HZ = 0.5 ms.
//
// --- Timing budget ---
// The SLM caps the system at 60 Hz (~16.7 ms/mask); the host waits
// settle_time (default 1 ms) after showing a mask, then reads the latest PD
// value. At 2000 frames/s a fresh frame arrives every 0.5 ms.
//
//   UART @ 115200 8N1: 115200 / 10 / 3 = 3840 frames/s max
//       -> 2000 frames/s uses ~52%, headroom for USB-CDC jitter.
//   ADC trigger: OUTPUT_RATE_HZ * SW_ACCUM_DEPTH = 8000 Hz (125 us period);
//       a 64x oversampled conversion at 1 Msps takes ~64 us (~51% duty).
//
// If you change ADC_HW_OVERSAMPLE / SW_ACCUM_DEPTH / OUTPUT_RATE_HZ, re-check
// both limits above.
//
// --- Hardware ---
//   Board:      EK-TM4C123GXL LaunchPad (TM4C123GH6PM), 80 MHz from PLL
//   Signal PD:  PE3 / AIN0 (ADC0, sequencer 3). Input range 0-3.3 V --
//               do NOT exceed 3.3 V on PE3 (scale/clamp the PD amp output).
//   UART:       UART0 (PA0/PA1) -> ICDI virtual COM port, 115200 8N1
//   Timer:      Timer0A periodic, hardware-triggers the ADC
//   Heartbeat:  green LED (PF3, the blinky LED) toggles every
//               HEARTBEAT_FRAMES frames -> ~1 Hz blink while streaming.
//
// --- Build (CCS) ---
//   Import TivaWare's examples/boards/ek-tm4c123gxl/blinky CCS project,
//   replace blinky.c with this main.c and startup_ccs.c with the one in this
//   folder (it routes the ADC Sequence 3 vector to ADC0SS3Handler; the stock
//   one sends it to IntDefaultHandler, which would hang the CPU). Build and
//   flash.
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/debug.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/adc.h"
#include "driverlib/uart.h"
#include "driverlib/timer.h"
#include "driverlib/interrupt.h"

// ---------------------------------------------------------------------------
// Tunable configuration
// ---------------------------------------------------------------------------
#define FRAME_START_BYTE    0xA5u

#define ADC_HW_OVERSAMPLE   64u     // hardware average per ADC read (2,4,8,16,32,64)
#define SW_ACCUM_DEPTH      4u      // software-averaged ADC reads per UART frame
#define OUTPUT_RATE_HZ      2000u   // frames/sec sent to the host

#define ADC_TRIGGER_HZ      (OUTPUT_RATE_HZ * SW_ACCUM_DEPTH)

#define UART_BAUD           115200u // must match photonic_annealing.py serial_baud

#define HEARTBEAT_FRAMES    (OUTPUT_RATE_HZ / 2u) // LED toggle -> 1 Hz blink

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
    while(1);
}
#endif

// ---------------------------------------------------------------------------
// State touched only by the ADC ISR (single producer, no locking needed).
// ---------------------------------------------------------------------------
static uint32_t g_accum = 0;
static uint32_t g_accumCount = 0;
static uint32_t g_heartbeatCount = 0;
static uint8_t g_ledState = 0;

// ---------------------------------------------------------------------------
// Enable a peripheral and wait until it is ready (same pattern as blinky).
// ---------------------------------------------------------------------------
static void
EnablePeripheral(uint32_t ui32Peripheral)
{
    SysCtlPeripheralEnable(ui32Peripheral);
    while(!SysCtlPeripheralReady(ui32Peripheral))
    {
    }
}

// ---------------------------------------------------------------------------
// Green LED on PF3, used as a streaming heartbeat (from blinky).
// ---------------------------------------------------------------------------
static void
ConfigureLED(void)
{
    EnablePeripheral(SYSCTL_PERIPH_GPIOF);
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_3);
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_3, 0);
}

// ---------------------------------------------------------------------------
// UART0 -> ICDI virtual COM port.
// ---------------------------------------------------------------------------
static void
ConfigureUART(void)
{
    EnablePeripheral(SYSCTL_PERIPH_UART0);
    EnablePeripheral(SYSCTL_PERIPH_GPIOA);

    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);
    GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    UARTConfigSetExpClk(UART0_BASE, SysCtlClockGet(), UART_BAUD,
                        (UART_CONFIG_WLEN_8 | UART_CONFIG_STOP_ONE |
                         UART_CONFIG_PAR_NONE));

    UARTFIFOEnable(UART0_BASE);
    UARTEnable(UART0_BASE);
}

// ---------------------------------------------------------------------------
// ADC0, sequencer 3: single channel (PE3/AIN0), hardware-oversampled,
// timer-triggered so the sampling cadence is exact and CPU-free.
// ---------------------------------------------------------------------------
static void
ConfigureADC(void)
{
    EnablePeripheral(SYSCTL_PERIPH_GPIOE);
    GPIOPinTypeADC(GPIO_PORTE_BASE, GPIO_PIN_3);

    EnablePeripheral(SYSCTL_PERIPH_ADC0);

    ADCHardwareOversampleConfigure(ADC0_BASE, ADC_HW_OVERSAMPLE);

    ADCSequenceDisable(ADC0_BASE, 3);
    ADCSequenceConfigure(ADC0_BASE, 3, ADC_TRIGGER_TIMER, 0);
    ADCSequenceStepConfigure(ADC0_BASE, 3, 0,
                             ADC_CTL_CH0 | ADC_CTL_IE | ADC_CTL_END);
    ADCSequenceEnable(ADC0_BASE, 3);

    ADCIntClear(ADC0_BASE, 3);
    ADCIntEnable(ADC0_BASE, 3);
    IntEnable(INT_ADC0SS3);
}

// ---------------------------------------------------------------------------
// Timer0A: periodic ADC trigger at ADC_TRIGGER_HZ. No ISR of its own --
// it only drives the ADC's trigger input.
// ---------------------------------------------------------------------------
static void
ConfigureTimer(void)
{
    EnablePeripheral(SYSCTL_PERIPH_TIMER0);

    TimerConfigure(TIMER0_BASE, TIMER_CFG_PERIODIC);
    TimerLoadSet(TIMER0_BASE, TIMER_A,
                 (SysCtlClockGet() / ADC_TRIGGER_HZ) - 1u);
    TimerControlTrigger(TIMER0_BASE, TIMER_A, true);
    TimerEnable(TIMER0_BASE, TIMER_A);
}

// ---------------------------------------------------------------------------
// ADC0 sequencer-3 ISR (hooked in startup_ccs.c): fires at ADC_TRIGGER_HZ.
// Accumulates hardware-oversampled reads; every SW_ACCUM_DEPTH-th one,
// averages and emits a 3-byte frame. The frame is only queued if the TX
// FIFO has room for all 3 bytes, so a stalled host drops whole frames
// rather than sending partial ones; the host resyncs on 0xA5 regardless.
// ---------------------------------------------------------------------------
void
ADC0SS3Handler(void)
{
    uint32_t sample;
    uint32_t avg;

    ADCIntClear(ADC0_BASE, 3);
    ADCSequenceDataGet(ADC0_BASE, 3, &sample); // 0-4095, hw-averaged

    g_accum += sample;
    g_accumCount++;

    if(g_accumCount < SW_ACCUM_DEPTH)
    {
        return;
    }

    avg = g_accum / SW_ACCUM_DEPTH; // still 0-4095
    g_accum = 0;
    g_accumCount = 0;

    // 16-byte TX FIFO: queue the frame only if it fits completely (the
    // FIFO-empty flag is too coarse, so try the first byte and only then
    // commit the rest -- at 52% UART load the FIFO never holds >2 frames).
    if(UARTCharPutNonBlocking(UART0_BASE, FRAME_START_BYTE))
    {
        UARTCharPut(UART0_BASE, (uint8_t)(avg & 0xFFu));
        UARTCharPut(UART0_BASE, (uint8_t)((avg >> 8) & 0x0Fu));
    }

    if(++g_heartbeatCount >= HEARTBEAT_FRAMES)
    {
        g_heartbeatCount = 0;
        g_ledState ^= GPIO_PIN_3;
        GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_3, g_ledState);
    }
}

int
main(void)
{
    // 80 MHz: 16 MHz XTAL, main osc, PLL (400 MHz / 2 / 2.5)
    SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN |
                   SYSCTL_XTAL_16MHZ);

    ConfigureLED();
    ConfigureUART();
    ConfigureADC();
    ConfigureTimer();

    IntMasterEnable();

    while(1)
    {
        // All work happens in ADC0SS3Handler.
    }
}
