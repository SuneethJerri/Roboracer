
# Lookahead Parameter Experiments

## 1. Objective

The objective of these experiments is to study how the Pure Pursuit
lookahead distance affects path-tracking accuracy and lap time.

Both normal lookahead sweeps and extreme-lookahead stress tests
were performed on the Spielberg track.

## 2. Lookahead Sweep

The base lookahead parameter was varied while keeping the other
controller parameters unchanged.

| Lookahead (m) | Lap time (s) | Maximum deviation (m) | Collisions |
|---:|---:|---:|---:|
| 0.50 | 173.5 | 0.0907 | 0 |
| 0.80 | 173.3 | 0.1373 | 0 |
| 1.10 | 173.2 | 0.2081 | 0 |
| 1.40 | 173.1 | 0.2826 | 0 |
| 1.70 | 172.8 | 0.3763 | 0 |
| 2.00 | 172.7 | 0.4796 | 0 |

### Observations

- Smaller lookahead values provide closer path tracking.
- Larger lookahead values slightly reduce lap time.
- Tracking deviation increases as the lookahead distance increases.
- All six configurations completed without collisions.

## 3. Stress Testing

Extreme lookahead values were tested to investigate controller
behaviour outside the normal operating range.

| Configuration | Base lookahead (m) | Lap time (s) | Max deviation (m) | Collisions |
|---|---:|---:|---:|---:|
| Very short | 0.10 | 173.55 | 0.0429 | 0 |
| Short | 0.25 | 173.55 | 0.0582 | 0 |
| Baseline | 0.80 | 173.35 | 0.1373 | 0 |
| Long | 2.50 | 172.30 | 0.6619 | 0 |
| Very long | 4.00 | 55.69 | 0.8480 | 1 |

The very-long configuration did not complete the lap and experienced
one collision.

## 4. Interpretation

A short lookahead produces more accurate geometric tracking but
requires more frequent steering adjustments.

A long lookahead produces smoother steering commands, but the vehicle
may cut corners and deviate farther from the reference path.

The 4.0 m configuration demonstrates a failure case, with a collision
before lap completion.

## 5. Baseline Selection

The 0.8 m base lookahead was retained as the baseline configuration.

It provides a practical compromise between tracking accuracy,
steering behaviour and lap time.

## 6. Conclusion

The experiments demonstrate that lookahead distance has a significant
effect on Pure Pursuit tracking behaviour.

Lookahead selection should balance path accuracy and smoothness rather
than optimizing lap time alone.

