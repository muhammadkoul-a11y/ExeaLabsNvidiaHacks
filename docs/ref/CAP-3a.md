# CAP-3a: scan intake tool

Lands one real scan (RoomPlan JSON, reference photos, three tape measurements) as `scans/<room_id>/` and checks it, so the 15 team scans arrive in one shape and without location metadata.

```bash
python scripts/intake_scan.py kitchen_01 --roomplan ~/Downloads/roomplan.json --photos a.jpg b.jpg \
    --location-type kitchen --phone "iPhone 15 Pro" --ios 18.1 --lighting daylight --clutter low \
    --measure counter_length=3.20,counter_height=0.92,sink_width=0.55
python scripts/intake_scan.py kitchen_01 --check   # after filling in a NO-GO folder
```

## What it writes

```text
scans/<room_id>/
  roomplan.json       # byte copy of the export
  reference_1.jpg ... # re-encoded: orientation applied, no EXIF at all (GPS, serial, timestamps, ICC), <= 1600 px, never upscaled
  notes.md            # NOTES_TEMPLATE in arrival/scan/intake.py, the CLI values filled in, the rest TODO
  measurements.json   # only when --measure is given: {"units": "meters", "source": "tape_measure", counter_length, counter_height, sink_width}
out/renders/<room_id>_roomplan_top.png
```

`measurements.json` is the machine-readable copy of the three tape values and the input TWN-1 reads for its 5 cm gate. `notes.md` holds the same numbers for people; keep them equal.

## Refusals (exit 2, nothing written)

- room id with anything but `[a-z0-9_]`
- `scans/<room_id>` already exists (intake never overwrites)
- fewer than 2 photos unless `--no-photos`, or a photo that is not an image
- a HEIC/HEIF photo (the iPhone camera's default): Pillow cannot read it, so the message says to convert it to JPEG or set Settings > Camera > Formats to Most Compatible before taking the photos
- a RoomPlan that does not load, or has fewer than 3 walls
- a measurement that is not a number of metres above 0 and at most 20 (centimetres typed by mistake), or an unknown key

The folder is built in `scans/.<room_id>.intake` and renamed at the end, so a failure leaves no half scan.

## Go / no-go (exit 0 / 1)

After writing, intake normalizes, compiles the twin, runs `place_default_objects`, and applies the `tests/test_scans.py` checks through its own helpers (`_overlap`, `_allowed`, `FLOOR_TOL`, `WALL_TOL`; that file is not changed): no overlapping fixture footprints, nothing below the floor, closed wall loop. It adds `validate_scan_dir`. It prints the fixture list and `GO` or `NO-GO` with each problem. A NO-GO folder is kept so the notes can be completed; `--check` (or `arrival.scan.intake.check(room_id)`) re-runs `validate_scan_dir` and the geometry checks on it, writes nothing, and exits 0 on GO, 1 on NO-GO, 2 when the folder does not exist.

`arrival.scan.intake.validate_scan_dir(path)` returns a list of problems, `[]` when the folder is complete: every notes.md field present and the three tape fields numbers of metres, at least 2 photos with no EXIF GPS tag, roomplan.json loads with 3 or more walls.

## Limits

- GO means structure and geometry pass, not metric accuracy; only the tape values compared against the twin (TWN-1) do that.
- Faces, screens and documents visible in the pixels are not detected; look at the photos.
- `tests/fixtures/roomplan/office.json` is NO-GO: its chair fixture overlaps the table footprint.
