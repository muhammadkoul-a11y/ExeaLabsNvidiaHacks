# Object detection fixtures (C4, TWN-5)

Synthetic only. `mug/` and `empty/` are CAP-2a RGB-D bundles (`frames_manifest.json`, RGB PNG, depth as 16-bit
millimetre PNG, ARKit camera axes) rendered by `python scripts/make_detection_fixture.py` from hero_kitchen's twin
with its objects removed. `mug/` adds one catalog mug (0.08 x 0.08 x 0.10 m) centred at (3.2, 0.45, 0.97) on
counter_b; `empty/` is the same three camera poses with nothing on the counter. 128 x 96 px, fx = fy = 110.

`replies.json` is hand-written, not a model reply. The tests keep these hand-written replies so they need no key; the live model's replies on the same frames are in
`docs/results/live-detect-2026-10-26.json`. Each mug
box is the mug's rendered pixel extent in that frame (the pixels whose depth differs from the empty room, as the
script prints them), with confidence 0.9; the empty frames list nothing. Entries are keyed by
`arrival.twin.detect.DetectionClient.key` of each RGB file, so re-rendering the frames changes the keys and the
replies must be rewritten.
