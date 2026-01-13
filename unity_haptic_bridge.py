"""
Unity Haptic Bridge - Sends haptic device data to Unity via TCP socket.
"""

import socket
import json
import time
from threading import Thread, Lock


class UnityHapticBridge:
    """Bridge to send haptic device data to Unity's SocketHapticDevice."""

    def __init__(self, host='127.0.0.1', port=5555, auto_connect=True):
        self.host = host
        self.port = port
        self.auto_connect = auto_connect

        self.server_socket = None
        self.client_socket = None
        self.connected = False
        self.running = False

        self._previous_position = [0.0, 0.0, 0.0]
        self._previous_time = time.time()
        self._lock = Lock()

        if auto_connect:
            self._start_server_thread()

    def _start_server_thread(self):
        """Start server in a background thread."""
        self.running = True
        self.server_thread = Thread(target=self._accept_connections, daemon=True)
        self.server_thread.start()

    def _accept_connections(self):
        """Accept incoming Unity client connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # Check running flag periodically

            print(f"[Unity Bridge] Listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()

                    with self._lock:
                        if self.client_socket:
                            self.client_socket.close()
                        self.client_socket = client_socket
                        self.connected = True

                    print(f"[Unity Bridge] Unity client connected from {client_address}")

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[Unity Bridge] Connection error: {e}")
                    break

        except Exception as e:
            print(f"[Unity Bridge] Server error: {e}")
        finally:
            self.connected = False

    def send_haptic_data(self, position, rotation, handle=0.0, clutch=0.0,
                        buttons=None, analog_inputs=None, dt=None):
        """
        Send haptic device data to Unity.

        Args:
            position: List of 3 floats [x, y, z]
            rotation: List of 4 floats [x, y, z, w] (quaternion)
            handle: Float value for handle/gripper state
            clutch: Float value for clutch state
            buttons: List of up to 6 bools for button states
            analog_inputs: List of up to 4 floats for analog inputs
            dt: Optional time delta for velocity calculation
        """
        if not self.connected or not self.client_socket:
            return False

        # Calculate velocity
        if dt is None:
            current_time = time.time()
            dt = current_time - self._previous_time
            self._previous_time = current_time

        velocity = [0.0, 0.0, 0.0]
        if dt > 0:
            velocity = [
                (position[i] - self._previous_position[i]) / dt
                for i in range(3)
            ]

        self._previous_position = position.copy()

        # Set defaults
        if buttons is None:
            buttons = [False] * 6
        if analog_inputs is None:
            analog_inputs = [0.0] * 4

        # Prepare data packet
        data = {
            "position": position,
            "rotation": rotation,
            "velocity": velocity,
            "handle": handle,
            "clutch": clutch,
            "buttons": buttons,
            "analogInputs": analog_inputs
        }

        try:
            json_data = json.dumps(data) + '\n'
            with self._lock:
                if self.client_socket:
                    self.client_socket.sendall(json_data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"[Unity Bridge] Send error: {e}")
            with self._lock:
                self.connected = False
                if self.client_socket:
                    try:
                        self.client_socket.close()
                    except:
                        pass
                    self.client_socket = None
            return False

    def is_connected(self):
        """Check if Unity client is connected."""
        with self._lock:
            return self.connected

    def close(self):
        """Close the bridge and all connections."""
        self.running = False

        with self._lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None

            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
                self.server_socket = None

            self.connected = False

        print("[Unity Bridge] Closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
