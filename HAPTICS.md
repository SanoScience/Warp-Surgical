# Haptics

This document explains the current haptics tuning panel and what each parameter does.

## Force Pipeline

The runtime force path is:

`contact reaction -> proxy target -> spring/damper force -> deadband -> low-pass -> max-force clamp -> slew-rate limit`

In plain terms:

- Tissue contact produces a reaction offset.
- That reaction offset moves a virtual proxy away from the real device pose.
- A spring-damper between the proxy and the real device generates force.
- Small/noisy force gets cleaned up by deadband and low-pass filtering.
- The final output is limited by `max_force` and `slew_rate_limit`.

The implementation lives in [omnisurg/runtime.py](/g:/warp/warp-surgical-dev/omnisurg/runtime.py#L944).

## Parameters

- `Enable Force Feedback`
  Turns force output on or off. If off, the device is commanded to zero force.

- `Reaction Scale`
  How much contact pushes the virtual proxy away from the real device position.
  Higher means contact creates a bigger virtual displacement, so force grows faster.
  If you feel almost nothing in contact, this is one of the first parameters to raise.

- `Proxy Follow`
  How quickly the virtual proxy moves toward its target.
  `1.0` means it snaps there immediately.
  Lower values make contact feel softer, laggier, and smoother.
  If contact feels mushy, raise this.

- `Max Proxy Offset`
  Hard cap on how far the virtual proxy can separate from the real device.
  Bigger values allow stronger spring stretch.
  Smaller values make the system softer and safer.

- `Spring K`
  The main stiffness term.
  It multiplies the proxy-device distance.
  Higher means a harder push-back for the same displacement.
  This is the main "make contact feel harder" parameter.

- `Damper B`
  The motion resistance term.
  It resists relative velocity between the proxy and the device.
  Higher means less buzzing and less oscillation, but it can feel heavier or sticky.
  If contact chatters, raise this a bit.

- `Deadband`
  Zeros out very small forces.
  Higher removes tiny noise, but it can also erase subtle contact.
  If light contact disappears, lower this or set it to `0`.

- `Lowpass Alpha`
  Smoothing amount on the force signal.
  `1.0` means almost no smoothing.
  Lower values add more smoothing and more lag.
  If force feels delayed or weak, raise this.
  If force feels noisy, lower it.

- `Max Force`
  Final cap on the commanded force magnitude.
  If this is too low, everything else can look correct but still feel weak.
  This is one of the first values to raise when debugging.

- `Slew Rate Limit`
  Limits how fast force is allowed to change over time.
  Higher means force can appear quickly.
  Lower means softer onset, but it can make the whole system feel muted.
  If force only appears after pushing for a while, this may be too low.

## Diagnostics

The haptics panel also shows these live readouts:

- `Contact Count`
  How many triangle contacts contributed this frame.

- `Avg Reaction Offset`
  The average opposite contact correction coming from collision.

- `Proxy Offset`
  The current gap between the virtual proxy and the real device.
  This is what the spring acts on.

- `Raw Force`
  Spring-damper output before cleanup.

- `Filtered Force`
  Force after deadband and low-pass.

- `Final Force`
  The force actually sent to the device after clamp and slew limiting.

- `Clamp Active`
  `yes` means `Max Force` is currently cutting the force down.

- `Slew Active`
  `yes` means `Slew Rate Limit` is currently slowing force changes.

## Quick Tuning Guide

- Too weak:
  Raise `Reaction Scale`, `Spring K`, and `Max Force`.
  Lower `Deadband`.
  Set `Lowpass Alpha` closer to `1.0`.
  Raise `Slew Rate Limit`.

- Too noisy:
  Raise `Damper B`.
  Lower `Lowpass Alpha`.
  Add a little `Deadband`.

- Too mushy:
  Raise `Proxy Follow` and `Lowpass Alpha`.

- Too sharp or jerky:
  Lower `Spring K` or `Reaction Scale`.
  Optionally lower `Slew Rate Limit`.

## Debug Recipe

If you want the clearest possible debug signal, start with:

- `Deadband = 0`
- `Lowpass Alpha = 1`
- `Proxy Follow = 1`
- `Slew Rate Limit = very high`
- `Max Force = high enough not to clamp`

Then watch `Raw Force` and `Final Force`:

- If `Raw Force` is large but `Final Force` is small, cleanup is suppressing the signal.
- If both are small, the contact signal itself is small.
