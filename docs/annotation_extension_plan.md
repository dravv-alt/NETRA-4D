# DrishtiAI annotation extension plan

This is a proposal only. It does not change Video 3 or its immutable CVAT XML.

## Unit of annotation

Annotate camera sessions as independent splits.  Draw boxes and stable track IDs for
people, phones, papers/chits, screens, hands when visible, and relevant desk regions.
For each person-frame record visibility (`visible`, `partially_occluded`, `occluded`,
`out_of_frame`) and an `unknown` option; absence of a box is never automatically a
negative.  Split development and held-out sets by session/camera, never random adjacent
frames.

## Attributes and temporal episodes

Add head direction and gaze target only when visually supportable, with values
`own_paper`, `neighbour`, `screen`, `environment`, `unknown`.  Add hand activity
(`writing`, `holding_object`, `reaching`, `hidden`, `unknown`) and object relations
(`near`, `held`, `reaching_toward`, `unknown`).  Mark episodes with start/end frame,
participants, visibility, annotator confidence, and ambiguity reason.  Separately label
normal head turns, low hands, conversations, invigilator interaction, occlusions, and
objects that resemble phones as hard negatives/review examples.

## Evaluation protocol

Evaluate detection boxes using only exhaustive labelled regions; report recall where
positive coverage exists and precision/mAP only where negatives are exhaustively covered.
Evaluate tracks with ID switches/fragmentation, pose with keypoint error or agreement to
labelled keypoints, and relations/episodes with independently reviewed windows.  Measure
inter-annotator agreement and retain ambiguous cases as `UNKNOWN`.  Behavioural states
remain evidence/review categories rather than cheating labels.
