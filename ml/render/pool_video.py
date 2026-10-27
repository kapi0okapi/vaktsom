"""A drowning, filmed: photorealistic video from two ceiling cameras, for an end-to-end test with
the real detector (frames -> detector -> tracking -> rules -> alarm).

Run inside Blender (one camera per call, frames split across jobs with --first/--last):

    blender -b --factory-startup -P ml/render/pool_video.py -- --out $DATA/video_demo --camera 0 --seed 3

A 25 m lap pool, 2 m deep, with swimmers in their lanes. One person struggles upright at the
surface (instinctive drowning response, bobbing), slides under, sinks and lies still on the bottom.
The water surface moves (animated waves). Writes JPEG frames, the camera calibration in the pool
frame (K, R, t in OpenCV convention) and the ground-truth timeline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pool_scene as ps  # noqa: E402

FPS = 10


def fcurves(obj) -> list:
    """F-curves of an object's action, for both the legacy and the layered (4.4+) action API."""
    act = obj.animation_data.action if obj.animation_data else None
    if act is None:
        return []
    if hasattr(act, "fcurves"):
        try:
            return list(act.fcurves)
        except Exception:
            pass
    out = []
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    return out


def set_interpolation(obj, mode: str) -> None:
    for fc in fcurves(obj):
        for kp in fc.keyframe_points:
            kp.interpolation = mode


def keyframe_visibility(obj, frames_on: tuple[int, int], last: int) -> None:
    keys = [(frames_on[0], False), (frames_on[1], True), (last + 1, True)]
    if frames_on[0] > 0:
        keys.insert(0, (0, True))
    for f, hidden in keys:
        obj.hide_render = hidden
        obj.keyframe_insert("hide_render", frame=f)
    set_interpolation(obj, "CONSTANT")


def build(seed: int, seconds: float, scenario: str = "active") -> dict:
    """scenario: active (struggle, then sink), passive (sudden sink), breath_hold (6 s on the bottom,
    then up again: must not alarm), normal (swimmers only: must not alarm)."""
    rng = random.Random(seed)
    ps.reset()
    sc = bpy.context.scene
    L, W = 25.0, 12.5
    depth = rng.choice([1.6, 2.0, 2.0, 3.0, 4.0])
    info = {"L": L, "W": W, "depth": depth}
    # reuse the hall builder with a fixed pool: patch the random choices
    orig = ps.random.Random

    class Fixed(random.Random):
        def choice(self, seq):
            if seq and isinstance(seq[0], float) and 12.5 in seq and 50.0 in seq:
                return L
            if seq and isinstance(seq[0], float) and 21.0 in seq:
                return W
            if seq and isinstance(seq[0], float) and 4.5 in seq:
                return depth
            return super().choice(seq)

    hall = ps.build_scene(Fixed(seed), argparse.Namespace())
    del orig
    for o in list(sc.objects):  # drop the random camera; we place two of our own
        if o.type == "CAMERA":
            bpy.data.objects.remove(o)
    # animated waves: move the displacement texture through the water
    tex = bpy.data.textures["waves"]
    empty = bpy.data.objects.new("wave_drive", None)
    sc.collection.objects.link(empty)
    water = next(o for o in sc.objects if o.modifiers.get("waves"))
    water.modifiers["waves"].texture_coords = "OBJECT"
    water.modifiers["waves"].texture_coords_object = empty
    water.modifiers["waves"].strength = rng.uniform(0.02, 0.05)
    tex.noise_scale = 0.6
    last = int(seconds * FPS)
    empty.location = (0, 0, 0)
    empty.keyframe_insert("location", frame=0)
    empty.location = (seconds * 0.35, seconds * 0.22, 0)
    empty.keyframe_insert("location", frame=last)
    set_interpolation(empty, "LINEAR")
    # swimmers in lanes (gliding at constant speed with a stroke pose)
    lanes = hall["lanes"] or [W * (k + 0.5) / 5 for k in range(5)]
    victim_lane = lanes[len(lanes) // 2]
    swimmers = []
    for k, y in enumerate(lanes):
        if y == victim_lane:
            continue
        body = ps.body_mesh(
            f"s{k}", rng.uniform(1.55, 1.9), rng.uniform(0.9, 1.2), rng.choice(["freestyle", "breaststroke"]), rng
        )
        body.data.materials.append(ps.body_material(rng))
        s = 1.0
        body.rotation_euler = Euler((math.radians(-90), 0, math.radians(90 if k % 2 else -90)), "XYZ")
        speed = rng.uniform(0.6, 1.1) * (1 if k % 2 else -1)
        x0 = rng.uniform(3, L - 3)
        for f in range(0, last + 1, 5):
            x = x0 + speed * f / FPS
            x = (x - 1.5) % (L - 3.0) + 1.5
            body.location = Vector((x, y, -0.12)) - (body.rotation_euler.to_matrix() @ Vector((0, 0, 1.0 * s)))
            body.keyframe_insert("location", frame=f)
        swimmers.append(body.name)
    # the victim
    vx, vy = rng.uniform(3.0, L - 3.0), victim_lane
    timeline = {"struggle_s": None, "submerged_s": None, "bottom_s": None, "surfaced_s": None}
    if scenario in ("active", "passive", "breath_hold"):
        t_sink = 8.0 if scenario == "active" else 3.0
        t_bottom = t_sink + (4.0 if scenario != "breath_hold" else 2.0)
        f_sink, f_bottom = int(t_sink * FPS), int(t_bottom * FPS)
        f_up = int((t_bottom + 6.0) * FPS) if scenario == "breath_hold" else last + 1
        mat = ps.body_material(rng)
        if scenario == "active":
            timeline["struggle_s"] = 0.0
            upright = ps.body_mesh("v_upright", 1.75, 1.05, "idr", rng)
            upright.data.materials.append(mat)
            upright.rotation_euler = (math.radians(-10), 0, math.radians(30))
            for f in range(0, f_sink + 1):
                bob = 0.12 * math.sin(2 * math.pi * 0.7 * f / FPS)
                upright.location = (vx, vy, -1.55 + bob - 0.1 * max(0, f - f_sink + 20) / 20)
                upright.keyframe_insert("location", frame=f)
            keyframe_visibility(upright, (0, f_sink), last)
        else:  # swimming normally until the moment it happens
            swim = ps.body_mesh("v_swim", 1.75, 1.05, "breaststroke", rng)
            swim.data.materials.append(mat)
            swim.rotation_euler = Euler((math.radians(-90), 0, math.radians(-90)), "XYZ")
            for f in range(0, f_sink + 1, 2):
                swim.location = Vector(
                    (vx - 0.5 * (f_sink - f) / FPS, vy, -0.12)
                ) - swim.rotation_euler.to_matrix() @ Vector((0, 0, 1.0))
                swim.keyframe_insert("location", frame=f)
            keyframe_visibility(swim, (0, f_sink), last)
        timeline["submerged_s"] = t_sink
        pose = "fetal" if scenario == "breath_hold" else "limp"
        limp = ps.body_mesh("v_limp", 1.75, 1.05, pose, rng)
        limp.data.materials.append(mat)
        for f in range(f_sink, f_bottom + 1):
            a = (f - f_sink) / max(f_bottom - f_sink, 1)
            rot = Euler((math.radians(-60 - 30 * a), 0, math.radians(30)), "XYZ")
            limp.rotation_euler = rot
            top_z = -0.4 - a * (depth - 0.55)
            limp.location = Vector((vx + 0.15 * a, vy, top_z)) - rot.to_matrix() @ Vector((0, 0, 1.0))
            limp.keyframe_insert("location", frame=f)
            limp.keyframe_insert("rotation_euler", frame=f)
        keyframe_visibility(limp, (f_sink, f_bottom), last)
        lying = ps.body_mesh("v_bottom", 1.75, 1.05, pose, rng)
        lying.data.materials.append(mat)
        lying.rotation_euler = Euler((math.radians(-90), 0, math.radians(30)), "XYZ")
        lying.location = Vector((vx + 0.15, vy, -depth + 0.14)) - lying.rotation_euler.to_matrix() @ Vector((0, 0, 1.0))
        keyframe_visibility(lying, (f_bottom, f_up), last)
        timeline["bottom_s"] = t_bottom
        if scenario == "breath_hold":  # comes up and swims away
            timeline["surfaced_s"] = f_up / FPS
            up = ps.body_mesh("v_up", 1.75, 1.05, "breaststroke", rng)
            up.data.materials.append(mat)
            up.rotation_euler = Euler((math.radians(-90), 0, math.radians(90)), "XYZ")
            for f in range(f_up, last + 1, 2):
                z = max(-depth + 0.4, -0.12 - 1.2 * max(0, (f_up + 20 - f)) / 20)
                up.location = Vector((vx + 0.6 * (f - f_up) / FPS, vy, z)) - up.rotation_euler.to_matrix() @ Vector(
                    (0, 0, 1.0)
                )
                up.keyframe_insert("location", frame=f)
            keyframe_visibility(up, (f_up, last + 1), last)
    return {
        "pool": info,
        "lanes": lanes,
        "scenario": scenario,
        "victim": {"x": vx + 0.15, "y": vy, **timeline},
        "swimmers": swimmers,
        "frames": last + 1,
        "fps": FPS,
    }


MOUNTS = [  # (position, target, horizontal fov) around a 25 x 12.5 m pool
    ((12.5, -3.0, 8.0), (12.5, 6.5, 0.0), 95.0),
    ((-3.0, 9.0, 8.5), (12.0, 6.0, 0.0), 90.0),
    ((12.5, 15.5, 7.0), (12.5, 6.0, 0.0), 95.0),
    ((28.0, 3.0, 9.0), (13.0, 6.5, 0.0), 90.0),
]


def cameras_for(seed: int) -> list:
    rng = random.Random(seed * 7 + 1)
    first = rng.randrange(len(MOUNTS))
    second = (first + rng.choice([1, 2, 3])) % len(MOUNTS)
    return [MOUNTS[first], MOUNTS[second]]


def add_camera(idx: int, width: int, height: int, seed: int) -> dict:
    sc = bpy.context.scene
    pos, target, fov = cameras_for(seed)[idx]
    data = bpy.data.cameras.new(f"cam{idx}")
    data.lens_unit = "FOV"
    data.angle = math.radians(fov)
    data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new(f"cam{idx}", data)
    sc.collection.objects.link(cam)
    cam.location = pos
    cam.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    bpy.context.view_layer.update()
    # OpenCV calibration in the pool frame: x right, y down, z forward
    R_bw = cam.matrix_world.to_3x3().transposed()
    flip = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    R = [[sum(flip[i][k] * R_bw[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    C = list(cam.matrix_world.translation)
    t = [-sum(R[i][j] * C[j] for j in range(3)) for i in range(3)]
    f = (width / 2) / math.tan(data.angle / 2)
    K = [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]
    return {
        "id": f"cam{idx}",
        "position": pos,
        "target": target,
        "hfov_deg": fov,
        "K": K,
        "R": R,
        "t": t,
        "size": [width, height],
    }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--scenario", default="active", choices=["active", "passive", "breath_hold", "normal"])
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--device", default="OPTIX")
    args = ap.parse_args(argv)
    truth = build(args.seed, args.seconds, args.scenario)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    dev = ps.gpu_setup(args.device)
    sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
    sc.render.resolution_x, sc.render.resolution_y = args.width, args.height
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 12
    sc.cycles.transmission_bounces = 12
    sc.cycles.blur_glossy = 0.5
    try:
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        sc.view_settings.view_transform = "Filmic"
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 90
    calib = add_camera(args.camera, args.width, args.height, args.seed)
    out = args.out / calib["id"]
    out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{calib['id']}.json").write_text(json.dumps(calib, indent=2))
    (args.out / "truth.json").write_text(json.dumps(truth, indent=2))
    last = truth["frames"] - 1 if args.last is None else min(args.last, truth["frames"] - 1)
    for f in range(args.first, last + 1):
        sc.frame_set(f)
        sc.render.filepath = str(out / f"{f:05d}.jpg")
        bpy.ops.render.render(write_still=True)
        print(f"frame {f} done", flush=True)


if __name__ == "__main__":
    main()
