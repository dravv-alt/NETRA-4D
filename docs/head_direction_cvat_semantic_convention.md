# CVAT head-direction semantic convention

## Status: confirmed by annotator

The Video 3 XML records the values `left`, `right`, `front`, `up`, `down`,
`back`, and occasionally unknown/missing variants. The annotator has now
confirmed that these labels are **student-relative**: directions describe the
student's own left/right/up/down/front, not image-facing or room-facing axes.
This statement comes from the annotation owner, not model-performance fitting.

Before a corrected head-direction evaluation is frozen, this convention must
be retained in the next immutable annotation export:

| Label | Required definition |
|---|---|
| LEFT | Student's own left |
| RIGHT | Student's own right |
| UP | Student's head pitched upward |
| DOWN | Student's head pitched downward |
| FRONT | Student facing forward in their own reference frame |

The current implementation uses an **image-coordinate proxy**: positive x is
`RIGHT`, negative x is `LEFT`, positive y is `DOWN`, and negative y is `UP`.
It must not be compared as though it already uses the student-relative CVAT
convention.

No CVAT XML or labels were modified by this document.
