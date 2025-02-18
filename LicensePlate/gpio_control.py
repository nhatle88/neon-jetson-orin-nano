import gpiod
import time

# Define the GPIO chip and line
chip = gpiod.Chip('gpiochip0')  # 'gpiochip0' is the default chip
line = chip.get_line(12)        # GPIO pin 12 (adjust as needed)

# Request the line as an output
line.request(consumer='gpio_control', type=gpiod.LINE_REQ_DIR_OUT)

try:
    while True:
        # Set the line high
        line.set_value(1)
        print("GPIO pin set to HIGH")
        time.sleep(1)

        # Set the line low
        line.set_value(0)
        print("GPIO pin set to LOW")
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting program (user interrupt).")
finally:
    # Release the line
    line.release()
