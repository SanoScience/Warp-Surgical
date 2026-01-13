#!/usr/bin/env python3
# Joystick/Gamepad HID monitor for Windows (Leonardo + HID-Project)
# Requires: pywinusb  ->  pip install pywinusb
#
# What it does:
#  - Finds the first HID device with Usage Page = Generic Desktop (0x01)
#    and Usage = Game Pad (0x05) or Joystick (0x04).
#  - Prints raw HID input reports as they arrive.
#  - Tries to parse the first 4 axes as 16-bit little-endian signed values
#    right after the report ID (common for NicoHood HID-Project gamepad).
#
# If parsing doesn't match your report layout, the raw hex dump displayed
# will still help adjust the offsets quickly.

import time
import struct
import sys

try:
    import pywinusb.hid as hid
except ImportError:
    print("Error: pywinusb is not installed. Install it with:\n  pip install pywinusb")
    sys.exit(1)

def find_game_controller():
    # Try Game Pad (usage=0x05)
    filter_gp = hid.HidDeviceFilter(usage_page=0x01, usage=0x05)
    devices = filter_gp.get_devices()
    if devices:
        return devices[0]
    # Fallback to Joystick (usage=0x04)
    filter_js = hid.HidDeviceFilter(usage_page=0x01, usage=0x04)
    devices = filter_js.get_devices()
    if devices:
        return devices[0]
    return None

def hex_bytes(data):
    return ' '.join(f'{b:02X}' for b in data)

def main():
    dev = find_game_controller()
    if not dev:
        print("No HID game controller found.\n"
              "Make sure your Arduino Leonardo is connected and recognized by Windows.\n"
              "Tip: Press Win+R -> joy.cpl to verify it's listed.")
        sys.exit(2)

    try:
        dev.open()
    except Exception as e:
        print("Failed to open HID device:", e)
        sys.exit(3)

    print("Connected to:")
    try:
        vid = f"0x{dev.vendor_id:04X}"
        pid = f"0x{dev.product_id:04X}"
    except Exception:
        vid, pid = "?", "?"

    print(f"  Vendor: {getattr(dev, 'vendor_name', '?')}  Product: {getattr(dev, 'product_name', '?')}  (VID={vid}, PID={pid})")
    print("\nStreaming input reports. Press Ctrl+C to quit.\n")

    def raw_handler(data):
        # data is a list of integers; data[0] is usually the Report ID
        report_id = data[0] if data else 0
        parsed = ""
        # Try to parse 4 axes (X,Y,Z,Rx) as <hhhh starting right after report_id
        if len(data) >= 9:
            payload = bytes(data[1:9])  # first 8 bytes after report id
            try:
                x, y, z, rx = struct.unpack('<hhhh', payload)
                parsed = f" | X={x:6d}  Y={y:6d}  Z={z:6d}  Rx={rx:6d}"
            except struct.error:
                pass

        print(f"RID={report_id:02d}  RAW=[{hex_bytes(data)}]{parsed}")

    # Set the handler for *all* input reports
    dev.set_raw_data_handler(raw_handler)

    try:
        # Keep process alive to receive events
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            dev.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
