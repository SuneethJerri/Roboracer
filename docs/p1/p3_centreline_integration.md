
# P3 Centreline Integration

## 1. Overview

The Pure Pursuit controller uses the P3-generated centreline as its
reference trajectory.

The centreline provides the geometric path that the vehicle must follow
during autonomous navigation.

## 2. Input Data

The controller loads the centreline from:

    results/p3_centreline/Spielberg_centreline.csv

The CSV contains path information including:

- x and y coordinates
- curvature
- cumulative path distance

These values are used for path projection, target selection and
curvature-based speed planning.

## 3. Path Projection

At each control step, the vehicle position is projected onto nearby
centreline segments.

The projection identifies the closest relevant point on the path and
helps estimate the vehicle's progress along the reference trajectory.

A bounded search window is used to avoid unnecessarily searching the
entire path at every step.

## 4. Target Point Selection

After estimating the vehicle's location on the centreline, the
controller searches ahead along the path to find a target point.

The target distance depends on the adaptive lookahead policy.

The target point is then transformed into the vehicle coordinate frame
before calculating the steering command.

## 5. Curvature Preview

The controller uses upcoming centreline curvature to estimate the
required speed.

The curvature preview helps the vehicle reduce speed before reaching
sharper turns.

## 6. Integration Pipeline

The complete control pipeline is:

1. Read vehicle pose and speed.
2. Project the vehicle position onto the P3 centreline.
3. Determine the lookahead distance.
4. Select a target point ahead on the centreline.
5. Transform the target into vehicle coordinates.
6. Calculate the Pure Pursuit steering angle.
7. Estimate upcoming curvature.
8. Calculate the desired speed.
9. Return the steering and speed commands.

## 7. Summary

The P3 centreline provides a shared geometric reference for the
Pure Pursuit controller.

Its integration enables path tracking and curvature-aware speed
planning without requiring the controller to generate a new path.
