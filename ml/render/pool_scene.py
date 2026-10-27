"""Photorealistic training and test images of indoor pools from ceiling cameras (Blender, Cycles).

Run inside Blender:

    blender -b --factory-startup -P ml/render/pool_scene.py -- --out $DATA/renders --n 400 --seed 0

Each image is a new random pool hall: basin size, depth (0.3 to 5 m), tiles and lane marks, water
with small waves (true refraction, IOR 1.333, and blue-green absorption in the volume), ceiling
lights and windows whose reflections become glare, and a camera 4 to 13 m up a wall or on the
ceiling. People are procedural (skin-modifier bodies in swimwear, all CC0 by construction) in
poses that matter for drowning detection: swimming and floating at the surface, standing in
shallow water, treading water upright, sinking, and lying still on the bottom.

Labels come from a second render of the same scene in which every body (and every head) emits a
unique flat colour and everything else is black, seen through the same refracting water: the boxes
are where the bodies appear in the camera, not where they are. Classes: person (head above the
surface), head, person_under (entirely below the surface).
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Vector

# --------------------------------------------------------------------------------------------- setup


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.objects,
    ):
        for item in list(coll):
            coll.remove(item)


def gpu_setup(prefer: str) -> str:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in [prefer, "OPTIX", "CUDA", "METAL", "HIP", "ONEAPI"]:
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                return kind
        except TypeError:
            continue
    return "CPU"


def principled(name: str, color, rough: float = 0.5, **inputs) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    for k, v in inputs.items():
        if k in b.inputs:
            b.inputs[k].default_value = v
    return m


def emission(name: str, color) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1.0)
    e.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], out.inputs[0])
    return m


def tile_material(rng: random.Random, name: str, base, line, scale: float) -> bpy.types.Material:
    """Square tiles with grout lines (brick texture with zero offset), slight colour noise."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexBrick")
    tex.offset = 0.0
    tex.inputs["Color1"].default_value = (*base, 1)
    tex.inputs["Color2"].default_value = (*[c * 0.94 for c in base], 1)
    tex.inputs["Mortar"].default_value = (*line, 1)
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Mortar Size"].default_value = 0.012
    tex.inputs["Brick Width"].default_value = 1.0
    tex.inputs["Row Height"].default_value = 1.0
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.25 + rng.random() * 0.2
    return m


# -------------------------------------------------------------------------------------------- bodies
# skeleton joints (metres) of an upright person facing +y, feet at z=0; scaled by height / 1.75
JOINTS = {
    "pelvis": (0, 0, 0.98),
    "spine": (0, 0, 1.18),
    "chest": (0, 0, 1.38),
    "neck": (0, 0, 1.55),
    "head": (0, 0, 1.68),
    "crown": (0, 0, 1.80),
    "l_shoulder": (-0.19, 0, 1.46),
    "l_elbow": (-0.22, 0, 1.18),
    "l_wrist": (-0.24, 0, 0.93),
    "l_hand": (-0.245, 0, 0.84),
    "r_shoulder": (0.19, 0, 1.46),
    "r_elbow": (0.22, 0, 1.18),
    "r_wrist": (0.24, 0, 0.93),
    "r_hand": (0.245, 0, 0.84),
    "l_hip": (-0.1, 0, 0.94),
    "l_knee": (-0.11, 0, 0.52),
    "l_ankle": (-0.11, 0, 0.09),
    "l_toe": (-0.11, 0.14, 0.03),
    "r_hip": (0.1, 0, 0.94),
    "r_knee": (0.11, 0, 0.52),
    "r_ankle": (0.11, 0, 0.09),
    "r_toe": (0.11, 0.14, 0.03),
}
RADIUS = {
    "pelvis": 0.15,
    "spine": 0.13,
    "chest": 0.16,
    "neck": 0.055,
    "head": 0.095,
    "crown": 0.07,
    "l_shoulder": 0.065,
    "l_elbow": 0.045,
    "l_wrist": 0.035,
    "l_hand": 0.04,
    "r_shoulder": 0.065,
    "r_elbow": 0.045,
    "r_wrist": 0.035,
    "r_hand": 0.04,
    "l_hip": 0.085,
    "l_knee": 0.06,
    "l_ankle": 0.04,
    "l_toe": 0.035,
    "r_hip": 0.085,
    "r_knee": 0.06,
    "r_ankle": 0.04,
    "r_toe": 0.035,
}
BONES = [
    ("pelvis", "spine"),
    ("spine", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("head", "crown"),
    ("chest", "l_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("l_wrist", "l_hand"),
    ("chest", "r_shoulder"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("r_wrist", "r_hand"),
    ("pelvis", "l_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("l_ankle", "l_toe"),
    ("pelvis", "r_hip"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
    ("r_ankle", "r_toe"),
]


def _rot_about(points: dict, names: list[str], pivot: str, euler: tuple) -> None:
    piv = Vector(points[pivot])
    R = Euler(euler, "XYZ").to_matrix()
    for n in names:
        points[n] = tuple(piv + R @ (Vector(points[n]) - piv))


def posed_joints(pose: str, rng: random.Random) -> dict:
    """Joint positions for a pose, still upright (the body is laid down by its placement)."""
    p = dict(JOINTS)
    j = lambda a: math.radians(a + rng.uniform(-12, 12))  # noqa: E731
    l_arm, r_arm = ["l_elbow", "l_wrist", "l_hand"], ["r_elbow", "r_wrist", "r_hand"]
    l_leg, r_leg = ["l_knee", "l_ankle", "l_toe"], ["r_knee", "r_ankle", "r_toe"]
    if pose == "freestyle":  # one arm forward over the head, one back
        _rot_about(p, l_arm, "l_shoulder", (j(-170), 0, 0))
        _rot_about(p, r_arm, "r_shoulder", (j(20), 0, 0))
        _rot_about(p, l_leg, "l_hip", (j(10), 0, 0))
        _rot_about(p, r_leg, "r_hip", (j(-10), 0, 0))
    elif pose == "breaststroke":
        _rot_about(p, l_arm, "l_shoulder", (j(-150), 0, j(30)))
        _rot_about(p, r_arm, "r_shoulder", (j(-150), 0, j(-30)))
        _rot_about(p, l_leg, "l_hip", (0, j(-25), 0))
        _rot_about(p, r_leg, "r_hip", (0, j(25), 0))
    elif pose in ("tread", "idr"):  # upright; IDR: arms out to the sides pressing down, head back
        spread = 75 if pose == "idr" else 45
        _rot_about(p, l_arm, "l_shoulder", (0, j(-spread), 0))
        _rot_about(p, r_arm, "r_shoulder", (0, j(spread), 0))
        _rot_about(p, l_leg, "l_hip", (j(-25), 0, 0))
        _rot_about(p, r_leg, "r_hip", (j(15), 0, 0))
        if pose == "idr":
            _rot_about(p, ["head", "crown"], "neck", (j(-30), 0, 0))
    elif pose == "limp":  # unconscious: arms and legs slightly bent, head hanging
        _rot_about(p, l_arm, "l_shoulder", (j(-40), 0, j(20)))
        _rot_about(p, r_arm, "r_shoulder", (j(-30), 0, j(-25)))
        _rot_about(p, ["l_ankle", "l_toe"], "l_knee", (j(35), 0, 0))
        _rot_about(p, ["r_ankle", "r_toe"], "r_knee", (j(20), 0, 0))
        _rot_about(p, ["head", "crown"], "neck", (j(30), 0, 0))
    elif pose == "fetal":
        _rot_about(p, l_leg, "l_hip", (j(-80), 0, 0))
        _rot_about(p, r_leg, "r_hip", (j(-80), 0, 0))
        _rot_about(p, ["l_ankle", "l_toe"], "l_knee", (j(90), 0, 0))
        _rot_about(p, ["r_ankle", "r_toe"], "r_knee", (j(90), 0, 0))
        _rot_about(p, l_arm, "l_shoulder", (j(-60), 0, 0))
        _rot_about(p, r_arm, "r_shoulder", (j(-60), 0, 0))
    elif pose == "stand":
        _rot_about(p, l_arm, "l_shoulder", (0, j(-15), 0))
        _rot_about(p, r_arm, "r_shoulder", (0, j(15), 0))
    return p


def body_mesh(name: str, height: float, build: float, pose: str, rng: random.Random) -> bpy.types.Object:
    joints = posed_joints(pose, rng)
    names = list(joints)
    s = height / 1.75
    verts = [Vector(joints[n]) * s for n in names]
    edges = [(names.index(a), names.index(b)) for a, b in BONES]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], edges, [])
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    sk = obj.modifiers.new("skin", "SKIN")
    sk.use_smooth_shade = True
    for i, n in enumerate(names):
        r = RADIUS[n] * s * (build if n in ("pelvis", "spine", "chest", "l_hip", "r_hip") else 1.0)
        me.skin_vertices[0].data[i].radius = (r, r * (0.8 if n in ("chest", "pelvis") else 1.0))
    me.skin_vertices[0].data[names.index("pelvis")].use_root = True
    sub = obj.modifiers.new("subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 2
    obj["joints"] = json.dumps({n: list(v) for n, v in zip(names, verts, strict=False)})
    return obj


SKIN_TONES = [
    (0.80, 0.62, 0.52),
    (0.72, 0.53, 0.42),
    (0.58, 0.42, 0.32),
    (0.42, 0.29, 0.21),
    (0.28, 0.19, 0.14),
    (0.88, 0.70, 0.60),
]


def body_material(rng: random.Random) -> bpy.types.Material:
    """Skin with a swimsuit band around the hips (and chest), by object-space height."""
    skin = rng.choice(SKIN_TONES)
    skin = tuple(min(1.0, c * rng.uniform(0.9, 1.1)) for c in skin)
    suit = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.3, 0.9), rng.uniform(0.1, 0.8))
    m = bpy.data.materials.new("body")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.45
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.15
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Object"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")  # height bands: legs skin, suit, torso (skin or suit), head
    lo, hi = rng.uniform(0.78, 0.86), rng.uniform(1.02, 1.08)
    full = rng.random() < 0.45
    top = rng.uniform(1.42, 1.48) if full else hi
    stops = [(0.0, skin), (lo, suit), (top, skin)]
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*skin, 1)
    els[1].position, els[1].color = lo / 2.0, (*suit, 1)
    e3 = els.new(top / 2.0)
    e3.color = (*skin, 1)
    hair = rng.choice(
        [
            (0.05, 0.04, 0.03),
            (0.25, 0.17, 0.08),
            (0.55, 0.42, 0.25),
            (0.8, 0.8, 0.8),
            colorsys.hsv_to_rgb(rng.random(), 0.8, 0.6),
        ]
    )  # last: a swim cap
    e4 = els.new(1.70 / 2.0)
    e4.color = (*hair, 1)
    scale = nt.nodes.new("ShaderNodeMath")
    scale.operation = "DIVIDE"
    scale.inputs[1].default_value = 2.0
    nt.links.new(sep.outputs["Z"], scale.inputs[0])
    nt.links.new(scale.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    del stops
    return m


# --------------------------------------------------------------------------------------------- scene
def build_scene(rng: random.Random, args) -> dict:
    scene = bpy.context.scene
    L = rng.choice([12.5, 16.0, 20.0, 25.0, 33.0, 50.0])
    W = rng.choice([6.0, 8.0, 10.0, 12.5, 15.0, 21.0, 25.0])
    W = min(W, L)
    depth = rng.choice([0.4, 0.9, 1.2, 1.4, 1.8, 2.0, 2.5, 3.5, 4.5, 5.0])
    # basin: floor and walls (inward-facing box without a lid), deck around it
    tiles = tile_material(
        rng,
        "tiles",
        colorsys.hsv_to_rgb(rng.uniform(0.5, 0.6), rng.uniform(0.05, 0.45), rng.uniform(0.75, 0.95)),
        (0.55, 0.6, 0.62),
        1 / 0.25 * 0.1,
    )
    deck = tile_material(
        rng,
        "deck",
        colorsys.hsv_to_rgb(rng.uniform(0.05, 0.12), rng.uniform(0.0, 0.25), rng.uniform(0.55, 0.85)),
        (0.4, 0.4, 0.4),
        4,
    )
    bpy.ops.mesh.primitive_cube_add(size=1, location=(L / 2, W / 2, -depth / 2))
    basin = bpy.context.object
    basin.scale = (L, W, depth)
    bpy.ops.object.transform_apply(scale=True)
    basin.data.polygons.foreach_set("use_smooth", [False] * len(basin.data.polygons))
    with_lid = [p for p in basin.data.polygons if p.normal.z > 0.9]
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(basin.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.9], context="FACES")
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(basin.data)
    bm.free()
    del with_lid
    basin.data.materials.append(tiles)
    # lane lines on the floor of lap pools
    lanes = []
    if L >= 20 and W >= 8:
        n = int(W // 2.5)
        line = principled("lane", (0.03, 0.06, 0.2), 0.4)
        for k in range(n):
            y = (k + 0.5) * W / n
            bpy.ops.mesh.primitive_plane_add(size=1, location=(L / 2, y, -depth + 0.003))
            o = bpy.context.object
            o.scale = (L - 4.0, 0.25, 1)
            o.data.materials.append(line)
            lanes.append(y)
    margin = 8.0
    bpy.ops.mesh.primitive_plane_add(size=1, location=(L / 2, W / 2, 0.0))
    d = bpy.context.object
    d.scale = (L + 2 * margin, W + 2 * margin, 1)
    bpy.ops.object.transform_apply(scale=True)
    bm = bmesh.new()
    bm.from_mesh(d.data)
    bmesh.ops.delete(bm, geom=bm.faces[:], context="FACES_ONLY")
    outer = [(-margin, -margin), (L + margin, -margin), (L + margin, W + margin), (-margin, W + margin)]
    inner = [(0, 0), (L, 0), (L, W), (0, W)]
    vo = [bm.verts.new((x, y, 0.0)) for x, y in outer]
    vi = [bm.verts.new((x, y, 0.0)) for x, y in inner]
    for k in range(4):
        bm.faces.new([vo[k], vo[(k + 1) % 4], vi[(k + 1) % 4], vi[k]])
    bm.to_mesh(d.data)
    bm.free()
    d.data.materials.append(deck)
    # lane ropes
    if lanes and rng.random() < 0.7:
        rope = principled("rope", colorsys.hsv_to_rgb(rng.choice([0.0, 0.6, 0.15]), 0.8, 0.7), 0.5)
        for k in range(1, len(lanes)):
            y = k * W / len(lanes)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.06, depth=L, location=(L / 2, y, 0.0), rotation=(0, math.pi / 2, 0), vertices=12
            )
            bpy.context.object.data.materials.append(rope)
    # water: displaced surface (refraction) and an absorbing volume
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=int(min(L, 50) * 6), y_subdivisions=int(min(W, 25) * 6), size=1, location=(L / 2, W / 2, 0.0)
    )
    water = bpy.context.object
    water.scale = (L - 0.02, W - 0.02, 1)
    bpy.ops.object.transform_apply(scale=True)
    tex = bpy.data.textures.new("waves", "CLOUDS")
    tex.noise_scale = rng.uniform(0.25, 0.9)
    disp = water.modifiers.new("waves", "DISPLACE")
    disp.texture = tex
    disp.strength = rng.uniform(0.005, 0.06)
    disp.mid_level = 0.5
    water.modifiers.new("smooth", "SUBSURF").render_levels = 1
    wmat = bpy.data.materials.new("water")
    wmat.use_nodes = True
    nt = wmat.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Roughness"].default_value = rng.uniform(0.0, 0.04)
    b.inputs["IOR"].default_value = 1.333
    b.inputs["Transmission Weight" if "Transmission Weight" in b.inputs else "Transmission"].default_value = 1.0
    vol = nt.nodes.new("ShaderNodeVolumeAbsorption")
    tint = colorsys.hsv_to_rgb(rng.uniform(0.47, 0.55), rng.uniform(0.1, 0.35), 1.0)
    vol.inputs["Color"].default_value = (*tint, 1)
    vol.inputs["Density"].default_value = rng.uniform(0.05, 0.35)
    nt.links.new(vol.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    water.data.materials.append(wmat)
    # close the water volume with a bottom face just above the floor, so absorption has a body
    bpy.ops.mesh.primitive_cube_add(size=1, location=(L / 2, W / 2, -depth / 2 - 0.001))
    wv = bpy.context.object
    wv.scale = (L - 0.02, W - 0.02, depth - 0.004)
    wv.data.materials.append(wmat)
    wv.hide_render = True  # kept for clarity; the surface plus absorption is enough in practice
    # light: ceiling panels and windows (their reflections on the water become glare)
    hall_h = rng.uniform(8, 15)
    height = hall_h - 0.3
    for _ in range(rng.randint(2, 5)):
        bpy.ops.object.light_add(type="AREA", location=(rng.uniform(0, L), rng.uniform(0, W), height))
        li = bpy.context.object.data
        li.size = rng.uniform(2, 6)
        li.energy = rng.uniform(800, 4000) * (L * W / 300) ** 0.5
        li.color = colorsys.hsv_to_rgb(rng.uniform(0.08, 0.15), rng.uniform(0.0, 0.25), 1.0)
    if rng.random() < 0.7:  # a row of windows along one long wall
        side = rng.choice([-1, 1])
        y = -margin + 0.05 if side < 0 else W + margin - 0.05
        win = emission("window", (1.0, 0.98, 0.95))
        win.node_tree.nodes["Emission"].inputs["Strength"].default_value = rng.uniform(3, 25)
        for _ in range(rng.randint(2, 6)):
            bpy.ops.mesh.primitive_plane_add(
                size=1, location=(rng.uniform(0, L), y, rng.uniform(3, 7)), rotation=(math.pi / 2, 0, 0)
            )
            o = bpy.context.object
            o.scale = (rng.uniform(2, 5), rng.uniform(2, 4), 1)
            o.data.materials.append(win)
    wall = principled(
        "wall",
        colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.35), rng.uniform(0.25, 0.85)),
        rng.uniform(0.4, 0.9),
    )
    ceiling = principled(
        "ceiling", colorsys.hsv_to_rgb(rng.uniform(0.05, 0.15), rng.uniform(0.0, 0.2), rng.uniform(0.2, 0.7)), 0.8
    )
    for loc, rot, sc_ in (
        ((L / 2, -margin, hall_h / 2), (math.pi / 2, 0, 0), (L + 2 * margin, hall_h, 1)),
        ((L / 2, W + margin, hall_h / 2), (math.pi / 2, 0, 0), (L + 2 * margin, hall_h, 1)),
        ((-margin, W / 2, hall_h / 2), (math.pi / 2, 0, math.pi / 2), (W + 2 * margin, hall_h, 1)),
        ((L + margin, W / 2, hall_h / 2), (math.pi / 2, 0, math.pi / 2), (W + 2 * margin, hall_h, 1)),
    ):
        bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
        o = bpy.context.object
        o.scale = sc_
        o.data.materials.append(wall)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(L / 2, W / 2, hall_h))
    o = bpy.context.object
    o.scale = (L + 2 * margin, W + 2 * margin, 1)
    o.data.materials.append(ceiling)
    world = bpy.data.worlds.new("hall")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.5, 0.52, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = rng.uniform(0.05, 0.4)
    scene.world = world
    # camera on a wall or under the ceiling, looking down into the basin
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(rng.uniform(55, 100))
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    h = min(rng.uniform(4, 13), hall_h - 0.6)  # below the ceiling
    side = rng.choice(["s", "n", "w", "e", "ceiling"])
    off = rng.uniform(0.5, 4)
    pos = {
        "s": (rng.uniform(0, L), -off, h),
        "n": (rng.uniform(0, L), W + off, h),
        "w": (-off, rng.uniform(0, W), h),
        "e": (L + off, rng.uniform(0, W), h),
        "ceiling": (rng.uniform(0, L), rng.uniform(0, W), h),
    }[side]
    target = Vector((rng.uniform(0.2, 0.8) * L, rng.uniform(0.2, 0.8) * W, 0))
    cam.location = pos
    direction = target - Vector(pos)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(rng.uniform(-8, 8)))
    scene.camera = cam
    return {"L": L, "W": W, "depth": depth, "camera": list(pos), "target": list(target), "lanes": lanes}


def place_people(rng: random.Random, info: dict, n: int) -> list[dict]:
    L, W, depth = info["L"], info["W"], info["depth"]
    people = []
    for k in range(n):
        height = rng.uniform(1.1, 1.95) if rng.random() < 0.8 else rng.uniform(0.85, 1.2)
        build = rng.uniform(0.85, 1.3)
        x, y = rng.uniform(0.8, L - 0.8), rng.uniform(0.8, W - 0.8)
        yaw = rng.uniform(0, 2 * math.pi)
        r = rng.random()
        if depth <= 1.35 and r < 0.45:
            state, pose = "stand", "stand"
        elif r < 0.55:
            state, pose = "surface", rng.choice(["freestyle", "breaststroke", "freestyle"])
        elif r < 0.65:
            state, pose = "float", "limp"
        elif r < 0.75 and depth > 1.35:
            state, pose = "tread", rng.choice(["tread", "tread", "idr"])
        elif r < 0.85:
            state, pose = "sinking", rng.choice(["limp", "fetal"])
        else:
            state, pose = "bottom", rng.choice(["limp", "limp", "fetal", "stand"])
        obj = body_mesh(f"p{k}", height, build, pose, rng)
        obj.data.materials.append(body_material(rng))
        s = height / 1.75
        if state == "stand":  # feet on the floor, head above water
            obj.rotation_euler = (0, 0, yaw)
            obj.location = (x, y, -depth)
            under = False
        elif state in ("surface", "float"):  # horizontal, back near the surface, head out
            face_up = state == "float" and rng.random() < 0.6
            obj.rotation_euler = Euler((math.radians(-90 + (180 if face_up else 0)), 0, yaw), "XYZ")
            obj.rotation_euler.rotate_axis("X", math.radians(rng.uniform(-12, 4)))
            obj.location = Vector((x, y, -0.12 * s)) - (obj.rotation_euler.to_matrix() @ Vector((0, 0, 1.0 * s)))
            under = False
        elif state == "tread":  # upright, head and shoulders out (IDR: mouth at the waterline)
            obj.rotation_euler = (math.radians(rng.uniform(-15, 15)), 0, yaw)
            sink = 1.55 if pose == "idr" else 1.42
            obj.location = (x, y, -sink * s)
            under = False
        elif state == "sinking":
            obj.rotation_euler = Euler((math.radians(rng.uniform(-120, -40)), math.radians(rng.uniform(-30, 30)), yaw))
            z = -rng.uniform(0.6, max(depth - 0.4, 0.7))
            obj.location = Vector((x, y, z)) - obj.rotation_euler.to_matrix() @ Vector((0, 0, 1.0 * s))
            under = True
        else:  # lying on the bottom (face down, on the back or curled up), or a standing child under water
            if pose == "stand" and depth > height + 0.2:
                obj.rotation_euler = (0, 0, yaw)
                obj.location = (x, y, -depth)
            else:
                face_down = rng.random() < 0.6
                obj.rotation_euler = Euler((math.radians(-90 if face_down else 90), 0, yaw), "XYZ")
                obj.location = Vector((x, y, -depth + 0.13 * s * build)) - (
                    obj.rotation_euler.to_matrix() @ Vector((0, 0, 1.0 * s))
                )
            under = True
        bpy.context.view_layer.update()
        top = max((obj.matrix_world @ Vector(c)).z for c in obj.bound_box)
        if not under and top < 0.02:
            under = True
        if under and top > -0.05:
            obj.location.z -= top + 0.08  # make "under" really under
        people.append({"name": obj.name, "state": state, "pose": pose, "under": under, "height": height})
    return people


# ---------------------------------------------------------------------------------------------- render
PALETTE = [colorsys.hsv_to_rgb(h / 24, 1.0, 1.0) for h in range(24)] + [
    colorsys.hsv_to_rgb(h / 24 + 1 / 48, 0.6, 0.7) for h in range(24)
]


def render_rgb(path: Path, args) -> None:
    sc = bpy.context.scene
    try:
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        sc.view_settings.view_transform = "Filmic"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 92
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_masks(path: Path, people: list[dict]) -> dict[str, tuple]:
    """Flat emission colours for bodies and heads, black elsewhere, same refracting water."""
    sc = bpy.context.scene
    black = emission("black", (0, 0, 0))
    keep = {"water"}
    swaps = []
    colours = {}
    for obj in sc.objects:
        if obj.type != "MESH":
            continue
        mats = obj.data.materials
        if obj.name.startswith("p") and obj.name[1:].isdigit():
            k = int(obj.name[1:])
            body_c = PALETTE[(2 * k) % len(PALETTE)]
            colours[obj.name] = body_c
            swaps.append((obj, list(mats)))
            mats.clear()
            mats.append(emission(f"m{k}", body_c))
            continue
        if any(m and m.name.startswith(tuple(keep)) for m in mats):
            # pure glass: same geometry, no absorption, so colours pass unchanged
            swaps.append((obj, list(mats)))
            glass = principled("glass", (1, 1, 1), 0.0)
            glass.node_tree.nodes["Principled BSDF"].inputs[
                "Transmission Weight"
                if "Transmission Weight" in glass.node_tree.nodes["Principled BSDF"].inputs
                else "Transmission"
            ].default_value = 1.0
            glass.node_tree.nodes["Principled BSDF"].inputs["IOR"].default_value = 1.333
            glass.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = (
                0.0 if "Specular IOR Level" in glass.node_tree.nodes["Principled BSDF"].inputs else 0.0
            )
            mats.clear()
            mats.append(glass)
            continue
        swaps.append((obj, list(mats)))
        mats.clear()
        mats.append(black)
    lights = [o for o in sc.objects if o.type == "LIGHT"]
    for li in lights:
        li.hide_render = True
    world_strength = sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sc.view_settings.view_transform = "Standard"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.render.filter_size = 0.01
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    # restore
    for obj, mats in swaps:
        obj.data.materials.clear()
        for m in mats:
            obj.data.materials.append(m)
    for li in lights:
        li.hide_render = False
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = world_strength
    sc.render.filter_size = 1.5
    return colours


def boxes_from_mask(path: Path, colours: dict[str, tuple], min_px: int) -> dict[str, tuple]:
    img = bpy.data.images.load(str(path))
    w, h = img.size
    px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    out = {}
    for name, c in colours.items():
        lin = np.array(c, np.float32)
        srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
        d = np.abs(px - srgb).max(-1)
        m = d < 0.08
        if m.sum() < min_px:
            continue
        ys, xs = np.nonzero(m)
        out[name] = (
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
            int(m.sum()),
        )
    return out


def head_boxes(people: list[dict]) -> dict[str, list[float]]:
    from bpy_extras.object_utils import world_to_camera_view

    sc = bpy.context.scene
    cam = sc.camera
    w, h = sc.render.resolution_x, sc.render.resolution_y
    out = {}
    for p in people:
        if p["under"]:
            continue
        obj = sc.objects[p["name"]]
        joints = json.loads(obj["joints"])
        centre = obj.matrix_world @ Vector(joints["head"])
        if centre.z < 0.02:
            continue
        r = 0.12 * p["height"] / 1.75
        pts = []
        for dx in (-r, r):
            for dy in (-r, r):
                for dz in (-r, r):
                    c = world_to_camera_view(sc, cam, centre + Vector((dx, dy, dz)))
                    if c.z <= 0:
                        break
                    pts.append((c.x * w, (1 - c.y) * h))
        if len(pts) < 8:
            continue
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        x0, y0, x1, y1 = max(min(xs), 0), max(min(ys), 0), min(max(xs), w - 1), min(max(ys), h - 1)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out[p["name"]] = [x0, y0, x1 - x0, y1 - y0]
    return out


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", default="OPTIX")
    ap.add_argument("--min-px", type=int, default=40)
    args = ap.parse_args(argv)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    labels = []
    for k in range(args.n):
        rng = random.Random(args.seed * 100_003 + k)
        reset()
        sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        dev = gpu_setup(args.device)
        sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
        sc.render.resolution_x, sc.render.resolution_y = args.width, args.height
        sc.render.resolution_percentage = 100
        sc.cycles.max_bounces = 12
        sc.cycles.transmission_bounces = 12
        sc.cycles.glossy_bounces = 4
        sc.cycles.blur_glossy = 0.5
        sc.cycles.caustics_refractive = True
        info = build_scene(rng, args)
        n_people = rng.randint(1, 12)
        people = place_people(rng, info, n_people)
        stem = f"r{args.seed:03d}_{k:05d}"
        render_rgb(args.out / "images" / f"{stem}.jpg", args)
        colours = render_masks(args.out / "masks" / f"{stem}.png", people)
        boxes = boxes_from_mask(args.out / "masks" / f"{stem}.png", colours, args.min_px)
        heads = head_boxes(people)
        anns = [{"category": "head", "bbox": b} for name, b in heads.items() if name in boxes]
        for p in people:
            b = boxes.get(p["name"])
            if b is None:
                continue
            anns.append(
                {
                    "category": "person_under" if p["under"] else "person",
                    "bbox": list(b[:4]),
                    "pixels": b[4],
                    "state": p["state"],
                    "pose": p["pose"],
                }
            )
        labels.append(
            {"file_name": f"{stem}.jpg", "width": args.width, "height": args.height, "scene": info, "annotations": anns}
        )
        (args.out / f"labels_{args.seed:03d}.json").write_text(json.dumps(labels))
        print(
            f"{stem}: {len(anns)} people ({sum(a['category'] == 'person_under' for a in anns)} under water)", flush=True
        )


if __name__ == "__main__":
    main()
