# SO-ARM101 robotic arm — vision-guided grasping

A complete control stack for a 5-axis arm running on a Raspberry Pi 5: a serial
driver written from scratch, inverse kinematics, hand-eye calibration, a
continuous control loop, and camera-guided grasping.

No vendor SDK, no robotics framework — everything is built up from the serial
protocol and the official URDF, so that every link in the chain is understood.

*(Le README en français est [ici](README.fr.md) — le code et les commentaires
sont en français.)*

<p align="center">
  <img src="media/suivi-doigt.gif" alt="The arm tracking a finger in real time" width="380">
</p>

<p align="center">
  <em>Live finger tracking: the camera locates the coloured marker, differential
  kinematics turn the error into a joint command, and a 50 Hz loop holds the
  gripper 1.5 cm ahead of it.</em><br>
  <a href="https://youtube.com/shorts/wYSlx46ZlL8">▶ Full demonstration (1 min)</a>
</p>

---

## What it does

An object sits on the table. The camera finds it, a homography turns its pixels
into millimetres in the arm's frame, inverse kinematics computes the joint
angles, and a 50 Hz control loop brings the gripper onto it without a single
jolt. On command, the jaws rotate perpendicular to the object and close on it.

The arm can also track a moving target live, holding the gripper a couple of
centimetres short of it.

---

## Hardware

| | |
|---|---|
| Arm | SO-ARM101, 5 axes + gripper, Feetech STS3215 servos |
| Computer | Raspberry Pi 5 |
| Camera | Camera Module 3 (imx708 sensor) |
| Link | half-duplex serial at 1 Mbaud, over a USB board |

---

## Architecture

The code is split so that no layer needs to know about the ones above it.

```
cinematique/     geometry, no hardware involved
  ik.py            forward and inverse kinematics (ikpy + official URDF)
  jacobienne.py    differential kinematics, for real-time tracking
  orientation.py   which wrist roll grasps a given object
  securite.py      is this motion acceptable?
  enveloppe.py     where can the gripper actually reach?

materiel/        the arm, real or simulated
  sts3215.py       serial driver, written from the SMS/STS protocol
  interface.py     the five questions you ask an arm
  servos.py        back-end driving the real servos
  simulation.py    back-end with no hardware, same refusals
  asservissement.py continuous 50 Hz control loop
  etalonnage.py    measured joint limits, URDF angle <-> servo degree mapping

vision/          the eyes
  camera.py        camera access, shared by everything else
  detection.py     colour tracking, position AND orientation
  etalonnage_camera.py  hand-eye calibration

apps/            applications
  suivi.py         tracks a target live, grasps it on command
```

What holds it together is `materiel/interface.py`: an abstract interface of five
methods. The vision and application code has no idea whether it is driving real
servos or a simulation, which makes it possible to write and exercise a whole
sequence without plugging the arm in.

---

## Problems worth solving

This project is mostly interesting for the faults that had to be found. Every
figure below is measured, not estimated.

**An axis the optimiser was picking at random.** The IK target here is a point,
with no orientation, and the wrist roll moves the gripper by only 1 mm per 10°.
To the optimiser it was therefore a free parameter, which it set arbitrarily:
three targets 1 cm apart produced +161°, +56° and +161° of roll, all within
0.00 mm of the target. The joint-travel guard then rejected most motions. The
roll was removed from the optimisation and given back its real job — orienting
the gripper.

**Jerky motion that came from the architecture.** The old controller split every
path into 10° steps and waited for arrival at each one, so the arm decelerated
to a standstill about ten times per move. Replacing that with a setpoint
refreshed at 50 Hz and never interrupted gave, on the same path, **8.91 s →
1.98 s**, with no intermediate stop at all.

**Fifteen serial round trips per step.** Commands went out register by register
and servo by servo, each waiting for its acknowledgement. The protocol's SYNC
WRITE instruction carries them all in a single frame: **15.00 ms → 0.25 ms**,
measured on the arm. All five axes now receive their setpoint within the same
millisecond.

**Inverse kinematics too slow to track.** The optimiser takes 18 ms median and
up to 328 ms worst case — hopeless for following a moving target. For a small
displacement, one forward-kinematics pass yields the Jacobian and the correction
falls out of a 3×3 solve: **231 µs, 45× faster**. As a bonus the method *cannot*
change the arm's configuration, which rules out 150° axis flips by construction.

**Two thirds of the sensor going unused.** The camera sensor exposes a cropped
mode that the library silently picks for 720p output. Forcing the full-sensor
mode widens the field of view by **a factor of 1.5 in each direction** — which
fixed a framing problem without touching the camera.

**An error measured honestly.** Hand-eye calibration reports 6.5 mm reprojection
error, but that figure is measured on the very points used to fit it.
Cross-validation gives **9.8 mm**, and that is the number that counts. More
importantly, the parallax from an object sitting 3 cm below the calibrated plane
is **27 to 30 mm** — three times the calibration error itself.

---

## Hand-eye calibration without a printed target

The camera says "object at pixel (712, 383)"; the arm works in metres. The usual
way to bridge the two is to print a checkerboard. Here, **the arm is its own
calibration target**: the gripper is driven to positions the model knows
exactly, a photograph is taken at each, and the gripper is clicked in every
image.

This is actually preferable to a printed target: that would give you the
camera ↔ target relationship, which still has to be tied back to the arm,
whereas this measures the camera ↔ arm relationship directly.

Since the objects to be grasped all rest on the table, the transformation sought
is a homography, fitted with RANSAC so that a missed click is discarded rather
than allowed to drag the result.

```bash
python -m vision.etalonnage_camera cadrage    # does the work area fit in frame?
python -m vision.etalonnage_camera capturer   # the arm walks a grid
python -m vision.etalonnage_camera pointer    # click the gripper in each image
python -m vision.etalonnage_camera ajuster    # homography + error
python -m vision.etalonnage_camera viser --aller   # click a spot, the arm goes
```

---

## Safety

Making an arm faster must not make it less careful.

- Every setpoint is bounded by **hand-measured joint limits**, which are safer
  than the URDF's: the elbow is declared there at +96.8° against a real stop at
  +47.7°.
- **Joint velocity and acceleration are bounded**, which removes jerk at source.
- **Cartesian floor**: no setpoint ever goes below 2.5 cm. Joint limits do
  nothing to stop the gripper being driven into the table.
- **Continuous load monitoring**, one axis per cycle: past the threshold,
  everything stops.
- **Emergency stop through a sentinel file**: `touch /tmp/bras-stop` freezes the
  arm from any session without touching the serial bus — two processes
  transmitting on a half-duplex line would garble each other.
- A target the arm cannot reach is **ignored**, not approached as closely as
  possible.

---

## Getting started

```bash
git clone https://github.com/QuentinMourier/bras-robot.git && cd bras-robot
python3 -m venv --system-site-packages venv
venv/bin/pip install -r requirements.txt

# No hardware at all: the simulated arm, same safeties, same refusals
venv/bin/python -m materiel.test_bras --simu etat
venv/bin/python -m materiel.simulation          # a full pick-and-place cycle
venv/bin/python comprendre_simulation.py        # how all of this works

# With the arm
venv/bin/python -m materiel.test_servos scan
venv/bin/python -m materiel.pilote              # keyboard jogging
venv/bin/python -m apps.suivi --couleur rouge   # tracking and grasping
```

`comprendre_simulation.py` is a runnable lesson in five steps: state, forward
kinematics, the simulation loop, the inverse problem, and what the simulation
cannot do.

---

## Known limitations

Stated plainly, because they decide what works and what does not.

- **The homography holds for one plane only.** A pixel denotes a ray, not a
  point; it is the calibrated plane that resolves the ambiguity. An object
  higher or lower is located off to the side, in proportion to that offset.
  Escaping this would take two cameras, a depth sensor, or calibration at two
  heights.
- **The simulation ignores physics.** No gravity, no contact, no torque: it
  validates geometry and logic, never safety. A trajectory validated in
  simulation can pass straight through the table.
- **No collision detection**, neither with the table nor of the arm with itself.
  Load monitoring is what protects, after the fact.
- **The gripper opens to about 3.5 cm.** Beyond that, an object can only be
  taken by a corner.
- **Some grasp orientations are unreachable** depending on posture: the roll
  turns about the forearm, which constrains the plane the jaws can open in. The
  code detects this and says so rather than closing at the wrong angle.

---

## Next

- Calibration at two heights, to locate an object whatever its thickness
- Learned detection instead of colour thresholding
- A fully autonomous grasp sequence, from spotting to placing

---

Code and comments are in French. Commit messages document the measurements and
the dead ends as much as the solutions — that is usually what you can no longer
reconstruct six months later.
