#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"

/* ML/TensorFlow includes */
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

/* Your IRIS model */
#include "models/IRISmodel.h"

#include <cstdarg>
#include <cstdio>

#define debug 1

/*****************************/
/* Global Variables          */
/*****************************/

bool led_blink_active_flag = true;
bool timer_interrupt_flag = false;
uint8_t uart_read_value;

/* TFLite globals */
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* model_input = nullptr;
    TfLiteTensor* model_output = nullptr;

    constexpr int kTensorArenaSize = 10 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

/*****************************/
/* Function Declarations     */
/*****************************/
void MicroPrintf(const char* format, ...) {
  char print_buf[256]; // Reasonably sized buffer for formatted string
  va_list args;
  va_start(args, format);
  vsnprintf(print_buf, sizeof(print_buf), format, args);
  va_end(args);
  printf("%s", print_buf); // Route to standard printf
}
void ml_init(void);
void timer_init(void);
static void isr_timer(void *callback_arg, cyhal_timer_event_t event);

// extern "C" required for C++ linkage
extern "C" void DebugLog(const char* s)
{
    printf("%s", s);
}

/*****************************/
/* MAIN Function             */
/*****************************/

int main(void)
{
	 cy_rslt_t result;

	#if defined (CY_DEVICE_SECURE)
	    cyhal_wdt_t wdt_obj;

	    /* Clear watchdog timer so that it doesn't trigger a reset */
	    result = cyhal_wdt_init(&wdt_obj, cyhal_wdt_get_max_timeout_ms());
	    CY_ASSERT(CY_RSLT_SUCCESS == result);
	    cyhal_wdt_free(&wdt_obj);
	#endif /* #if defined (CY_DEVICE_SECURE) */

	    /* Initialize the device and board peripherals */
	    result = cybsp_init();

	    /* Board init failed. Stop program execution */
	    if (result != CY_RSLT_SUCCESS)
	    {
	        CY_ASSERT(0);
	    }

	    /* Enable global interrupts */
	    __enable_irq();

    /* Initialize UART */
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
            CYBSP_DEBUG_UART_CTS,CYBSP_DEBUG_UART_RTS,CY_RETARGET_IO_BAUDRATE);
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    printf("\x1b[2J\x1b[;H"); // Clear terminal
    printf("**************************************************************\r\n");
    printf("*   PSoC 6 TensorFlow Lite IRIS Classification Demo          *\r\n");
    printf("**************************************************************\r\n\n");

    /* Initialize ML */
    ml_init();

    printf("Running IRIS classification...\r\n");
    printf("Press 'Enter' key to pause or resume the demo\r\n\n");

    float user_input[4];
    const char* iris_classes[] = {"Setosa", "Versicolor", "Virginica"};

    for (;;)
    {
        /* Check UART for pause/resume */
        if (cyhal_uart_getc(&cy_retarget_io_uart_obj, &uart_read_value, 1) == CY_RSLT_SUCCESS)
        {
            if (uart_read_value == '\r')
            {
                if (led_blink_active_flag)
                {
                    printf("Demo paused \r\n");
                }
                else
                {
                    printf("Demo resumed\r\n");
                }
                printf("\x1b[1F");
                led_blink_active_flag ^= 1;
            }
        }

        if (led_blink_active_flag)
        {
            /* Prompt user to enter 4 features */
            printf("\r\nEnter 4 features for IRIS classification:\r\n");

            printf("Sepal Length (cm): ");
            scanf("%f", &user_input[0]);

            printf("Sepal Width (cm): ");
            scanf("%f", &user_input[1]);

            printf("Petal Length (cm): ");
            scanf("%f", &user_input[2]);

            printf("Petal Width (cm): ");
            scanf("%f", &user_input[3]);

            /* Feed user input into model input tensor */
            for (int i = 0; i < 4; i++) {
                model_input->data.f[i] = user_input[i];
            }

            /* Run inference */
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) {
                printf("Invoke failed on user input\r\n");
                continue; // Skip to next iteration
            }

            /* Get output probabilities */
            float* output = model_output->data.f;
            int output_len = model_output->dims->data[model_output->dims->size - 1];

            /* Find class with highest probability */
            int max_index = 0;
            float max_prob = output[0];
            for (int i = 1; i < output_len; i++) {
                if (output[i] > max_prob) {
                    max_prob = output[i];
                    max_index = i;
                }
            }

            /* Print result */
            printf("Input: [%.2f, %.2f, %.2f, %.2f] -> Predicted Class: %s (%.3f)\r\n",
                   user_input[0], user_input[1], user_input[2], user_input[3],
                   iris_classes[max_index], max_prob);

            /* Optional delay */
            cyhal_system_delay_ms(500);
        }

        if (timer_interrupt_flag)
        {
            timer_interrupt_flag = false;
        }
    }
}

/*****************************/
/* ML Initialization         */
/*****************************/

void ml_init(void)
{
    printf("Initializing TensorFlow Lite Micro...\r\n");

    model = tflite::GetModel(IRISmodel_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model version does not match Schema version\r\n");
        CY_ASSERT(0);
    }

    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    // Add any other ops your model needs

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors() failed\r\n");
        CY_ASSERT(0);
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

#if DEBUG
    printf("Model loaded successfully\r\n");
    printf("Input tensor shape: [");
    for (int i = 0; i < model_input->dims->size; i++) {
        printf("%d", model_input->dims->data[i]);
        if (i < model_input->dims->size - 1) printf(", ");
    }
    printf("]\r\n");

    printf("Output tensor shape: [");
    for (int i = 0; i < model_output->dims->size; i++) {
        printf("%d", model_output->dims->data[i]);
        if (i < model_output->dims->size - 1) printf(", ");
    }
    printf("]\r\n");

    printf("TensorFlow Lite Micro initialization complete\r\n\n");
#endif
}

/*****************************/
/* Timer Stub                */
/*****************************/

void timer_init(void)
{
     //You can leave this empty
}

static void isr_timer(void *callback_arg, cyhal_timer_event_t event)
{
    timer_interrupt_flag = true;
}

/* [] END OF FILE */

