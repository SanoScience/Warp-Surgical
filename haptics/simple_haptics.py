import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice
from dataclasses import dataclass

# Data class to keep track of the device state and use it in other parts of the code
@dataclass
class DeviceState:
    # Define the variables you want to safe here

# Callback to gather the device state
@hd_callback
def device_callback():
    # Make the device_state global to be accesed in other parts of the code
    global device_state
    # your callback function, gather the different variables on the device
    # YOUR CODE HERE

if __name__ == "__main__":
    # Initialize the data class
    device_state = DeviceState()

    # Initialize the haptic device and the callback loop
    device = HapticDevice(callback = device_callback, scheduler_type="async")
    
    # YOUR CODE HERE
    
    # Close the device to avoid segmentation faults
    device.close()