# Query recipes for body-pose-vlm

All coordinates are normalized to `[0, 1000]` as `[x, y]`. End every query with
the normalization sentence (the model was trained with it).

## Full skeleton of one person

```
Report all visible human body keypoints of the leftmost person in the image.
Use COCO keypoint names, coordinates normalized to 0-1000 as [x, y].
```
Variants: `rightmost`, `second from the left`, `#3 from the left (middle)`, `person`.

## One arm (shoulder–elbow–wrist)

```
What are the keypoint coordinates of the right arm (shoulder, elbow, wrist)
of the second from the left person? Coordinates normalized to 0-1000 as [x, y].
```

## Wrist only (grasp targeting)

```
Where is the left hand of the rightmost person? Give the wrist keypoint.
Coordinates normalized to 0-1000.
```

## Attribute / action references (zero-shot, experimental)

These were never in training data but generally work on clean scenes:

```
Report the keypoints of the right hand (shoulder, elbow, wrist) of the person
with the longest hair. Coordinates normalized to 0-1000 as [x, y].

One person is looking toward the person on his left side. Report the keypoints
of the left arm (shoulder, elbow, wrist) of the person being looked at.
Coordinates normalized to 0-1000 as [x, y].

The person is holding a tennis racket. Report only the keypoints of the arm
that holds the racket (shoulder, elbow, wrist). Coordinates normalized to
0-1000 as [x, y].
```

Chinese prompts also work (answers still come back in the English JSON protocol):

```
只输出中间那个人的左手手臂的关键点
```

## Known-bad patterns (for now)

- `"…for every person…"` multi-target enumeration → returns one person only.
  Workaround: one request per ordinal reference and merge client-side.
- Requesting joints outside COCO-17 (fingers, toes) → not trained yet.
