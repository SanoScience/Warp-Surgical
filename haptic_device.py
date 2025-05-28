from dataclasses import dataclass, field
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
import time

@dataclass
class DeviceState:
    button: bool = False
    position: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    gimbals: list = field(default_factory=list)
    force: list = field(default_factory=list)

class HapticController:
    def __init__(self, device_name="Default Device", scale=2.5):
        self.device_state = DeviceState()
        self.device_state.force = [0.0, 0.0, 0.0]
        self.scale = scale
        
        # Set up callback
        self._setup_callback()
        
        # Initialize device
        self.device = HapticDevice(device_name=device_name, callback=self._state_callback)
        time.sleep(0.2)  # Allow device to initialize
    
    def _setup_callback(self):
        @hd_callback
        def state_callback():
            transform = hd.get_transform()
            joints = hd.get_joints()
            gimbals = hd.get_gimbals()
            self.device_state.position = [transform[3][0], transform[3][1], transform[3][2]]
            self.device_state.joints = [joints[0], joints[1], joints[2]]
            self.device_state.gimbals = [gimbals[0], gimbals[1], gimbals[2]]
            hd.set_force(self.device_state.force)
            button = hd.get_buttons()
            self.device_state.button = True if button == 1 else False
        
        self._state_callback = state_callback
    
    def get_scaled_position(self):
        """Get the current haptic device position scaled for the simulation."""
        return [pos * self.scale for pos in self.device_state.position]
    
    def set_force(self, force):
        """Set the force feedback to the haptic device."""
        self.device_state.force = force
    
    def is_button_pressed(self):
        """Check if the haptic device button is pressed."""
        return self.device_state.button
