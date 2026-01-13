import time
import argparse
from haptic_device import HapticController
from unity_haptic_bridge import UnityHapticBridge


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Unity haptic bridge with real device")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Unity server host")
    parser.add_argument("--port", type=int, default=5555, help="Unity server port")
    parser.add_argument("--scale", type=float, default=1.0, help="Position scale factor")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("="*60)
    print("Unity Haptic Bridge Test")
    print("="*60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Scale: {args.scale}")
    print(f"Duration: {args.duration}s")
    print("="*60)

    # Initialize haptic controller
    print("\nInitializing haptic device...")
    try:
        haptic = HapticController(scale=args.scale)
        print("✓ Haptic device initialized")
    except Exception as e:
        print(f"✗ Failed to initialize haptic device: {e}")
        return

    # Initialize Unity bridge
    print(f"\nStarting Unity bridge server on {args.host}:{args.port}...")
    print("Waiting for Unity to connect...")

    with UnityHapticBridge(host=args.host, port=args.port, auto_connect=True) as bridge:
        start_time = time.time()
        last_status_time = start_time
        frame_count = 0

        try:
            while (time.time() - start_time) < args.duration:
                # Get haptic device state
                position = haptic.get_scaled_position()
                rotation = haptic.get_rotation()
                button = haptic.is_button_pressed()

                # Prepare data
                buttons = [button] + [False] * 5
                handle = 1.0 if button else 0.0

                # Send to Unity
                success = bridge.send_haptic_data(
                    position=position,
                    rotation=rotation,
                    handle=handle,
                    clutch=0.0,
                    buttons=buttons
                )

                frame_count += 1

                # Print status every second
                current_time = time.time()
                if current_time - last_status_time >= 1.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed

                    if bridge.is_connected():
                        print(f"[{elapsed:.1f}s] ✓ Connected | "
                              f"Pos: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] | "
                              f"Rot: [{rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}, {rotation[3]:.2f}] | "
                              f"Button: {button} | "
                              f"FPS: {fps:.1f}")
                    else:
                        print(f"[{elapsed:.1f}s] ✗ Waiting for Unity connection...")

                    last_status_time = current_time

                # Small delay to avoid overloading
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

        finally:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print("\n" + "="*60)
            print("Test Summary")
            print("="*60)
            print(f"Duration: {elapsed:.2f}s")
            print(f"Frames: {frame_count}")
            print(f"Average FPS: {fps:.1f}")
            print(f"Final Connection Status: {'Connected' if bridge.is_connected() else 'Disconnected'}")
            print("="*60)


if __name__ == "__main__":
    main()
