# Later — do not build tonight

Specified so the next night does not redesign. None of these are tasks in [06-agent-tasks.md](06-agent-tasks.md).

## Physics generator

A native prototype is now available in `arrival/generate/se3_retarget.py` (see [the command and provenance contract](REFERENCE.md#closed-loop-eval)). It retargets executed end-effector trajectories in object and sink SE(3) frames, bridges segments smoothly, solves IK and exports only successful MuJoCo rollouts without human contact. Its generator label is `object_centric_se3_retarget`; it does not call the upstream MimicGen framework. G7/issue #62 remains open for licensed or teleoperated seed sets, actual upstream integration, the 100-episode benchmark and comparison against the expert. The external import probe and S20 translation replay remain separate.

Replace `scripted_retarget` with RoboCasa MimicGen for `counter_to_sink`:

```bash
git clone https://github.com/robocasa-benchmark/mimicgen
git clone https://github.com/ARISE-Initiative/robomimic -b robocasa
python mimicgen/scripts/prepare_src_dataset.py --dataset <seed.hdf5>
python mimicgen/scripts/generate_dataset_multicore.py \
  --source <seed.hdf5> \
  --config mimicgen/exps/templates/robocasa/single_stage/kitchen_pnp/PnPCounterToSink.json
```

Seeds are either RoboCasa's human demonstrations for that task or about ten teleoperated episodes in the compiled twin. Map successful MimicGen trajectories into the same LeRobot contract. Set `joints_are_placeholder` to false only when `single_arm` holds real joint positions. Set `arrival_generator` to `mimicgen` and `arrival_physically_valid` to true.

`counter_to_counter` needs a real template name from the RoboCasa task list, or ten teleop seeds and a custom MimicGen config. Until then its template stays `null`.

The oracle stays: drop an attempt when the final object is more than 0.05 m horizontally from the target, or the grasp never closed. Failures are not relabeled into chat JSONL.

Wire `arrival.generate.mimicgen` to call this. Tonight that module only reports whether the import works.

## Splat composite

Train a Gaussian splat on scan frames that include camera poses. Implement the depth composite in [03-architecture.md](03-architecture.md). Set `arrival_appearance` to `splat_composite`. If the layers do not line up, fall back to texturing the simulator meshes from the scan photos, and record `arrival_appearance = "mesh_texture"`.

## Cosmos

After physically valid frames exist, send a share of them through the Cosmos 3 Generator endpoint. Start at 10 percent of frames. Store both the original and the restyled frame ids in episode metadata. This is a variation axis that was explicitly taken off the critical path.

## People

Do not add people to task JSON. A later variation axis `people_present` can be added beside lighting. Driving them is a Token Factory call with Nemotron 3.5 Lightning, plus a crowd in the simulator. Omitted until pick-and-place export is physically valid.

## Drawers and doors

The spec's stretch goal. The splat cannot show a drawer opening, so these tasks wait until the appearance path can show moving fixtures. Not a compiler feature for tonight beyond leaving cabinet types in the layout.

## Gap filling

Nemotron 3 Ultra is the spec's tool for occluded fixture geometry. The compiler does not guess unseen counters except for the explicit sink/stove synthesis rule.

## Scale-out

One scan should fan out to many Serverless Jobs that shut down when the episode batch is done. Checkpoint partial datasets to Object Storage so a preempted job can resume. The local queue's `queued` status is the stand-in for that job record.

## Real robot

Out of scope. The first real-robot test happens after this hackathon, in the scanned room, using a policy trained on a physically valid dataset.

## Scan replacement

Drop a real RoomPlan JSON and RGB-D frames with poses into `scans/hero_kitchen/` and re-run `demo`. Delete the authored sample only when the normalizer's twin matches the measured room within a tolerance you write down. Layout classes to support if the mapping gets messy: straight, L, and galley. Show three clean scans before claiming a general floor plan.

## Fine-tune

Only after `arrival_physically_valid` is true and joints are real. Command is in the README from task T13. Modality config keys must match `meta/modality.json`: video `ego_view`, state and action `single_arm` and `gripper`, language `annotation.human.task_description`. Hardware: 40 GB+ VRAM. Fallback policy: the LeRobot cookbook job.
