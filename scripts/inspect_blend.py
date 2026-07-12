"""Inspect Perseverance .blend for Isaac Sim readiness."""
import bpy
from mathutils import Vector

objs = list(bpy.data.objects)

print("===ARMATURE===")
arm = bpy.data.objects.get("Armature")
if arm:
    print("bones:", [b.name for b in arm.data.bones])
    for b in arm.data.bones:
        parent = b.parent.name if b.parent else None
        print(
            f"  bone {b.name} head={tuple(round(x, 4) for x in b.head_local)} "
            f"tail={tuple(round(x, 4) for x in b.tail_local)} parent={parent}"
        )
    for pb in arm.pose.bones:
        cons = [(c.type, c.name) for c in pb.constraints]
        if cons:
            print(f"  pose {pb.name} constraints={cons}")
    # which meshes are armature-deformed?
    for o in objs:
        for m in o.modifiers:
            if m.type == "ARMATURE":
                print(f"  deformed: {o.name} -> {m.object.name if m.object else None}")

print("===WHEELS===")
w = bpy.data.objects.get("Wheels_objs")
if w and w.type == "MESH":
    print(
        f"single mesh verts={len(w.data.vertices)} polys={len(w.data.polygons)} "
        f"dim={tuple(round(x, 4) for x in w.dimensions)} "
        f"loc={tuple(round(x, 4) for x in w.location)}"
    )
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(w.data)
    bm.verts.ensure_lookup_table()
    visited = set()
    sizes = []
    for v in bm.verts:
        if v.index in visited:
            continue
        stack = [v]
        visited.add(v.index)
        n = 0
        while stack:
            cur = stack.pop()
            n += 1
            for e in cur.link_edges:
                ov = e.other_vert(cur)
                if ov.index not in visited:
                    visited.add(ov.index)
                    stack.append(ov)
        sizes.append(n)
    sizes.sort(reverse=True)
    print(f"loose_parts={len(sizes)} top_sizes={sizes[:12]}")
    # centroids of large parts
    visited = set()
    centroids = []
    for v in bm.verts:
        if v.index in visited:
            continue
        stack = [v]
        visited.add(v.index)
        verts = []
        while stack:
            cur = stack.pop()
            verts.append(cur.co.copy())
            for e in cur.link_edges:
                ov = e.other_vert(cur)
                if ov.index not in visited:
                    visited.add(ov.index)
                    stack.append(ov)
        if len(verts) < 500:
            continue
        c = sum(verts, Vector()) / len(verts)
        centroids.append((len(verts), tuple(round(x, 3) for x in c)))
    centroids.sort(reverse=True)
    print("large_part_centroids (count, xyz):")
    for item in centroids[:10]:
        print(" ", item)
    bm.free()

print("===SUSPENSION===")
s = bpy.data.objects.get("suspension")
if s:
    nverts = len(s.data.vertices) if s.type == "MESH" else None
    print(
        f"type={s.type} verts={nverts} dim={tuple(round(x, 4) for x in s.dimensions)} "
        f"children={[c.name for c in s.children]}"
    )
    if s.type == "MESH":
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(s.data)
        visited = set()
        sizes = []
        for v in bm.verts:
            if v.index in visited:
                continue
            stack = [v]
            visited.add(v.index)
            n = 0
            while stack:
                cur = stack.pop()
                n += 1
                for e in cur.link_edges:
                    ov = e.other_vert(cur)
                    if ov.index not in visited:
                        visited.add(ov.index)
                        stack.append(ov)
            sizes.append(n)
        sizes.sort(reverse=True)
        print(f"suspension loose_parts={len(sizes)} top_sizes={sizes[:15]}")
        bm.free()

print("===CAMERA_NAMED===")
for o in sorted(objs, key=lambda x: x.name):
    n = o.name.lower()
    if any(k in n for k in ("cam", "haz", "nav", "mast")):
        parent = o.parent.name if o.parent else None
        print(
            f"{o.name} type={o.type} parent={parent} "
            f"dim={tuple(round(x, 3) for x in o.dimensions)} "
            f"loc={tuple(round(x, 3) for x in o.location)}"
        )

print("===WORLD_BOUNDS===")
mn = Vector((1e9,) * 3)
mx = Vector((-1e9,) * 3)
for o in objs:
    if o.type != "MESH":
        continue
    for corner in o.bound_box:
        wco = o.matrix_world @ Vector(corner)
        mn.x = min(mn.x, wco.x)
        mn.y = min(mn.y, wco.y)
        mn.z = min(mn.z, wco.z)
        mx.x = max(mx.x, wco.x)
        mx.y = max(mx.y, wco.y)
        mx.z = max(mx.z, wco.z)
size = mx - mn
print(
    f"min={tuple(round(x, 4) for x in mn)} max={tuple(round(x, 4) for x in mx)} "
    f"size={tuple(round(x, 4) for x in size)}"
)
print(
    f"XY footprint ~ {size.x:.3f} x {size.y:.3f}, height {size.z:.3f} "
    "(real Perseverance ~3.0 x 2.7 x 2.2 m)"
)

print("===PHYSICS===")
rb = sum(1 for o in objs if getattr(o, "rigid_body", None))
rbc = sum(1 for o in objs if getattr(o, "rigid_body_constraint", None))
print(f"rigid_bodies={rb} rb_constraints={rbc}")
print(f"actual CAMERA objects={sum(1 for o in objs if o.type == 'CAMERA')}")
print(f"EMPTY objects={[o.name for o in objs if o.type == 'EMPTY']}")

print("===PARENTING FLATNESS===")
roots = [o for o in objs if o.parent is None]
print(f"root_count={len(roots)} (many roots = not a single articulated tree)")

print("===KEY_COLLECTIONS===")
for c in bpy.data.collections:
    if any(k in c.name.lower() for k in ("wheel", "haz", "nav", "susp", "mast", "cam", "body")):
        print(c.name, "->", [o.name for o in c.objects][:25])
