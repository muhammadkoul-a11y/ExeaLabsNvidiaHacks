# RoomPlan exports (test fixtures)

Apple RoomPlan `CapturedRoom` JSON exports used for normalizer, compiler, sim, and intake tests.

| file | description | licence | source |
| --- | --- | --- | --- |
| `kitchen_real_01.json` | Real kitchen capture with flush cabinets and appliances | Apache 2.0 | Consented teammate capture (P2, `docs/SCANNING.md`) |
| `office.json` | Office capture: table, chairs, storage, doors (open/closed), window, polygon floor | Apache 2.0 | Authored test fixture (replaces unlicensed `amal.json`, task P5) |
| `study.json` | Study / bedroom capture: desk blocked by chair, bed, television, storage, closed door | Apache 2.0 | Authored test fixture (replaces unlicensed `shopyseek.json`, task P5) |

## Licence & Provenance (Task P5)

Previously, two third-party exports (`amal.json` and `shopyseek.json`) without software licenses were temporarily used as test fixtures. Under hackathon rules and open-source standards, ungranted third-party assets cannot be redistributed.

Both fixtures were replaced with clean, authored captures under Apache 2.0:
- `office.json`: exercises non-kitchen table interactions (`move_on_table`), solid furniture, walkers, room-appropriate objects, portal rendering, and intake overlap validation.
- `study.json`: exercises bedroom category classification (`room_kind`), portal rendering, and the edge-case where a desk's free long edge is blocked by a chair so no task can be placed.

### Real-device coverage note
`kitchen_real_01.json` is a real phone capture, but contains no doors or windows. `office.json` and `study.json` are clean authored test fixtures with axis-aligned geometry and round dimensions. Real phone captures of an office and bedroom with doors and windows remain open under task P5 to validate portal placement against real-device sensor noise and arbitrary yaw angles.

These fixtures are used by:
- `tests/test_intake.py`
- `tests/test_non_kitchen.py`
- `tests/test_roomplan.py`
- `tests/test_roomplan_coverage.py`
- `tests/test_summary.py`
- `tests/test_openings_sim.py`
- `tests/test_ios_bundle.py`
- `tests/test_web.py`
- `arrival/eval/summary.py`
- `scripts/render_rooms.py`
