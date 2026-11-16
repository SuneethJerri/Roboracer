

\# Pure Pursuit Controller Implementation



\## 1. Overview



Pure Pursuit is a geometric path-tracking algorithm used for autonomous

vehicle navigation. It calculates the steering angle required to guide

the vehicle towards a target point located ahead on the reference path.



The controller uses the vehicle's current position and orientation,

together with a lookahead target on the reference centreline.



\## 2. Steering Geometry



Let the target point in the vehicle coordinate frame be:



\- x: longitudinal distance to the target

\- y: lateral distance to the target



The distance to the target is:



&#x20;   Ld = sqrt(x^2 + y^2)



The curvature required to reach the target is:



&#x20;   kappa = 2\*y / Ld^2



For a vehicle with wheelbase L, the steering angle is:



&#x20;   delta = atan(L \* kappa)



Combining the equations:



&#x20;   delta = atan(2\*L\*y / (x^2 + y^2))



This is the Pure Pursuit steering law.



\## 3. Adaptive Lookahead



The implementation uses a speed-dependent lookahead distance:



&#x20;   Ld = clip(Ld\_base + k\_v \* max(0, speed),

&#x20;             Ld\_min, Ld\_max)



Where:



\- Ld\_base is the base lookahead distance.

\- k\_v is the speed coefficient.

\- speed is the current vehicle speed.

\- Ld\_min and Ld\_max are the minimum and maximum lookahead limits.



A larger lookahead generally produces smoother steering, while a

smaller lookahead allows closer tracking of the reference path.



\## 4. Path Tracking



The controller uses the P3 centreline as its reference path.



At each control step, it:



1\. Finds the vehicle's projection onto the reference path.

2\. Identifies a target point at the adaptive lookahead distance.

3\. Transforms the target point into the vehicle coordinate frame.

4\. Calculates the steering angle using the Pure Pursuit equation.

5\. Estimates upcoming path curvature to determine the desired speed.



\## 5. Curvature-Based Speed Control



The desired speed is limited using the lateral acceleration constraint:



&#x20;   v = min(speed\_cap,

&#x20;           sqrt(a\_lat\_max / max(abs(kappa), 1e-6)))



Where:



\- speed\_cap is the maximum allowed speed.

\- a\_lat\_max is the lateral acceleration limit.

\- kappa is the preview curvature.



This reduces speed in sharper turns.



\## 6. Current Configuration



The current configuration includes:



| Parameter | Value |

|---|---:|

| Base lookahead | 0.8 m |

| Speed coefficient | 0.2 |

| Minimum lookahead | 0.5 m |

| Maximum lookahead | 3.0 m |

| Speed cap | 2.0 m/s |

| Lateral acceleration limit | 2.0 m/s^2 |

| Wheelbase | 0.33 m |



The wheelbase is provisional and should be verified against the

simulator vehicle model.



\## 7. Implementation Limitations



\- Performance depends on the accuracy of the reference centreline.

\- Lookahead selection affects tracking accuracy and steering smoothness.

\- The wheelbase must match the simulated vehicle.

\- AutoDRIVE validation is still pending.



\## 8. Summary



The implemented controller combines geometric Pure Pursuit steering,

adaptive lookahead selection, P3 centreline tracking, and

curvature-based speed control.



