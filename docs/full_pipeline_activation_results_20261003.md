# Video 3 full experimental activation result

Run: `outputs/experimental/video03_full_experimental_sampled_v2_20261003`.
The run is a frozen-baseline augmentation replay over all 2,200 canonical Video 3
frames, not a replacement for release inference.  It preserved the baseline output
hash and did not read, write, or alter the CVAT XML.

Enabled: RTMPose, MediaPipe hand evidence, interaction graph, temporal evidence,
hypotheses, explicit evidence states, evidence fusion, and the guarded 3D component.
RTMPose/MediaPipe were sampled every 100 frames because ONNX Runtime exposes only a
CPU provider in this environment.  Every non-sampled hand record is `NOT_SAMPLED`,
not a negative or an inferred absence.

Observed outputs: 3,031 person-frame evidence records; 365 mutually confident pose
keypoint pairs with mean normalized inter-model distance 0.03088; 23 hand landmark
observations; 5,327 baseline object-association geometry records; 1,476 neighbour
proximity geometry records; and zero `REVIEW_REQUIRED` records.  The last result is
not a negative behavioural finding: the guarded review transition requires multiple
independent observable streams and none met that rule in this sampled experiment.

The Multi-HMR checkpoint is recorded with its hash and four representative cases, but
the 3D records are `PENDING_ADAPTER_VALIDATION`: there is no validated conversion from
its mesh/camera coordinates to these tracked people, so no orientation values were
invented.  Hand, interaction, temporal, and behavioural accuracy remain unmeasured
because their compatible labels do not yet exist.  No precision, F1, mAP, false-positive
rate, or cheating probability is reported.
