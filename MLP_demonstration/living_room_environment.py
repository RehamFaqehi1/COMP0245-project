"""Visual-only living-room dressing shared by the MLP task GUIs."""


def _visual_box(client, half_extents, position, rgba, orientation=None):
    visual_id = client.createVisualShape(
        client.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgba,
    )
    return client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=orientation or [0.0, 0.0, 0.0, 1.0],
    )


def _visual_plant(client, x, y):
    """A small decorative plant assembled from non-colliding visuals."""
    pot = client.createVisualShape(
        client.GEOM_CYLINDER,
        radius=0.17,
        length=0.25,
        rgbaColor=[0.68, 0.25, 0.13, 1.0],
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=pot,
        basePosition=[x, y, 0.125],
    )

    leaf_colors = [
        [0.16, 0.42, 0.20, 1.0],
        [0.20, 0.52, 0.24, 1.0],
        [0.12, 0.34, 0.17, 1.0],
    ]
    for dx, dy, dz, radius, color in (
        (0.0, 0.0, 0.43, 0.20, leaf_colors[0]),
        (0.15, 0.02, 0.52, 0.17, leaf_colors[1]),
        (-0.12, 0.08, 0.55, 0.18, leaf_colors[2]),
        (0.0, -0.14, 0.61, 0.16, leaf_colors[1]),
    ):
        leaf = client.createVisualShape(
            client.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=leaf,
            basePosition=[x + dx, y + dy, dz],
        )


def add_living_room_environment(client, ground_body=None):
    """Add wood flooring, walls, windows, a clock, desk, and door.

    Every added room object is visual-only. The simulator's original ground
    plane remains the only environment collision surface.
    """
    if ground_body is not None:
        client.changeVisualShape(
            ground_body,
            -1,
            rgbaColor=[0.72, 0.55, 0.38, 1.0],
        )

    # Larger wood floor, centered around the original robot and task targets.
    _visual_box(
        client,
        [6.0, 6.0, 0.004],
        [0.5, 1.0, -0.003],
        [0.55, 0.31, 0.16, 1.0],
    )
    seam_color = [0.30, 0.16, 0.08, 1.0]
    for x in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        _visual_box(client, [0.006, 6.0, 0.001], [x, 1.0, 0.0015], seam_color)
    for y in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        _visual_box(client, [6.0, 0.006, 0.001], [0.5, y, 0.0015], seam_color)

    wall = [0.88, 0.83, 0.73, 0.48]
    trim = [0.36, 0.22, 0.13, 1.0]
    wall_z = 1.30
    wall_height = 1.30
    wall_thickness = 0.06

    # Front wall and right wall with a doorway opening from y=-0.65 to y=0.55.
    _visual_box(client, [6.0, wall_thickness, wall_height], [0.5, -5.0, wall_z], wall)
    _visual_box(client, [wall_thickness, 2.175, wall_height], [6.5, -2.825, wall_z], wall)
    _visual_box(client, [wall_thickness, 3.225, wall_height], [6.5, 3.775, wall_z], wall)
    _visual_box(client, [0.10, 0.60, 0.10], [6.45, -0.05, 2.50], wall)
    _visual_box(client, [0.06, 0.06, 1.10], [6.44, -0.65, 1.10], trim)
    _visual_box(client, [0.06, 0.06, 1.10], [6.44, 0.55, 1.10], trim)

    # Open door hinged at the lower jamb and swung into the room.
    door_orientation = client.getQuaternionFromEuler([0.0, 0.0, -1.5708])
    _visual_box(
        client,
        [0.035, 0.53, 1.02],
        [5.97, -0.65, 1.02],
        [0.42, 0.22, 0.11, 1.0],
        door_orientation,
    )
    knob_id = client.createVisualShape(
        client.GEOM_SPHERE,
        radius=0.045,
        rgbaColor=[0.80, 0.62, 0.20, 1.0],
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=knob_id,
        basePosition=[5.45, -0.55, 1.02],
    )

    # Framed whiteboard opposite the TV/monitor, facing into the room.
    board_x = 1.25
    board_y = -4.85
    board_z = 1.55
    _visual_box(
        client,
        [1.10, 0.035, 0.70],
        [board_x, -4.90, board_z],
        [0.28, 0.18, 0.10, 1.0],
    )
    _visual_box(
        client,
        [1.03, 0.012, 0.63],
        [board_x, board_y, board_z],
        [0.97, 0.97, 0.94, 1.0],
    )
    # This orientation keeps the writing upright and facing into the room.
    board_text_orientation = [0.0, 0.7071068, 0.7071068, 0.0]
    text_lines = (
        ("Group 19", 1.88, 0.42),
        ("Yingxu", 1.60, 0.34),
        ("Reham", 1.32, 0.34),
        ("Abivarshan", 1.04, 0.34),
    )
    for text, z, size in text_lines:
        client.addUserDebugText(
            text,
            [2.12, -4.825, z],
            textColorRGB=[0.08, 0.08, 0.08],
            textSize=size,
            lifeTime=0,
            textOrientation=board_text_orientation,
        )

    # Back wall surrounds two window openings rather than covering them.
    back_y = 7.0
    _visual_box(client, [6.0, wall_thickness, 0.50], [0.5, back_y, 0.50], wall)
    _visual_box(client, [6.0, wall_thickness, 0.40], [0.5, back_y, 2.20], wall)
    mid_z = 1.40
    mid_half_height = 0.40
    for x_center, half_width in ((-3.675, 1.825), (-0.15, 0.30), (4.025, 2.475)):
        _visual_box(
            client,
            [half_width, wall_thickness, mid_half_height],
            [x_center, back_y, mid_z],
            wall,
        )

    # Window panes, frames, sills, and mullions.
    window_centers_x = (-1.15, 0.85)
    glass = [0.45, 0.72, 0.88, 0.62]
    frame = [0.30, 0.19, 0.11, 1.0]
    for x in window_centers_x:
        _visual_box(client, [0.64, 0.012, 0.35], [x, 6.93, 1.40], glass)
        for x_edge in (x - 0.68, x + 0.68):
            _visual_box(client, [0.035, 0.035, 0.40], [x_edge, 6.91, 1.40], frame)
        for z_edge in (0.98, 1.82):
            _visual_box(client, [0.71, 0.035, 0.035], [x, 6.91, z_edge], frame)
        _visual_box(client, [0.014, 0.025, 0.38], [x, 6.90, 1.40], frame)
        _visual_box(client, [0.64, 0.025, 0.014], [x, 6.90, 1.40], frame)
        # Curtain rail and folded panels at either side of each window.
        _visual_box(client, [0.84, 0.025, 0.025], [x, 6.84, 1.91], trim)
        curtain = [0.34, 0.48, 0.63, 0.88]
        for side in (-1, 1):
            _visual_box(
                client,
                [0.12, 0.035, 0.42],
                [x + side * 0.78, 6.82, 1.40],
                curtain,
            )

    # Simple round wall clock on the solid right portion of the back wall.
    clock_orientation = client.getQuaternionFromEuler([1.5708, 0.0, 0.0])
    clock_face = client.createVisualShape(
        client.GEOM_CYLINDER,
        radius=0.22,
        length=0.045,
        rgbaColor=[0.96, 0.94, 0.85, 1.0],
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=clock_face,
        basePosition=[2.65, 6.92, 1.55],
        baseOrientation=clock_orientation,
    )
    _visual_box(client, [0.008, 0.075, 0.006], [2.65, 6.885, 1.59], [0.12, 0.10, 0.08, 1.0], clock_orientation)
    _visual_box(client, [0.075, 0.008, 0.006], [2.68, 6.885, 1.55], [0.12, 0.10, 0.08, 1.0], clock_orientation)

    # A small desk tucked beneath the left window.
    desk_wood = [0.43, 0.25, 0.13, 1.0]
    _visual_box(client, [0.82, 0.34, 0.045], [-4.15, 6.15, 0.78], desk_wood)
    for x in (-4.82, -3.48):
        for y in (5.89, 6.41):
            _visual_box(client, [0.045, 0.045, 0.37], [x, y, 0.37], desk_wood)
    _visual_box(client, [0.32, 0.045, 0.22], [-4.15, 6.37, 1.04], [0.12, 0.16, 0.20, 1.0])
    _visual_box(client, [0.22, 0.012, 0.13], [-4.15, 6.32, 1.05], [0.42, 0.70, 0.82, 1.0])

    # Bookcase on the same wall as the whiteboard, spaced to its left.
    bookshelf = [0.43, 0.25, 0.14, 1.0]
    _visual_box(client, [0.72, 0.24, 0.96], [-3.55, -4.69, 0.96], bookshelf)
    for shelf_z in (0.36, 0.92, 1.48):
        _visual_box(client, [0.68, 0.28, 0.035], [-3.55, -4.42, shelf_z], [0.58, 0.39, 0.22, 1.0])
    book_colors = ([0.55, 0.12, 0.10, 1.0], [0.12, 0.28, 0.52, 1.0], [0.82, 0.63, 0.20, 1.0], [0.18, 0.42, 0.25, 1.0])
    for row_z in (0.58, 1.14):
        for index, color in enumerate(book_colors):
            _visual_box(
                client,
                [0.045, 0.08, 0.20 + 0.025 * (index % 2)],
                [-4.00 + index * 0.14, -4.42, row_z],
                color,
            )

    # TV and console closer to the sofa, facing it across the coffee table.
    tv_x = 4.55
    tv_y = 2.10
    _visual_box(client, [0.95, 0.34, 0.27], [tv_x, tv_y, 0.30], desk_wood)
    for x in (tv_x - 0.72, tv_x + 0.72):
        _visual_box(client, [0.045, 0.30, 0.28], [x, tv_y, 0.30], trim)
    # Lower the TV so its bottom edge sits directly on the console top (z=0.57).
    _visual_box(client, [0.78, 0.055, 0.45], [tv_x, tv_y - 0.34, 1.02], [0.07, 0.08, 0.09, 1.0])
    _visual_box(client, [0.70, 0.012, 0.37], [tv_x, tv_y - 0.275, 1.02], [0.16, 0.30, 0.40, 1.0])

    # Small decorative vase on the console, beside the TV rather than in its screen.
    vase = client.createVisualShape(
        client.GEOM_CYLINDER,
        radius=0.12,
        length=0.30,
        rgbaColor=[0.72, 0.34, 0.22, 1.0],
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vase,
        basePosition=[tv_x + 0.82, tv_y, 0.72],
    )
    vase_neck = client.createVisualShape(
        client.GEOM_CYLINDER,
        radius=0.055,
        length=0.16,
        rgbaColor=[0.76, 0.40, 0.27, 1.0],
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vase_neck,
        basePosition=[tv_x + 0.82, tv_y, 0.95],
    )

    # Living-room rug, sofa, and coffee table, all safely outside the task zone.
    _visual_box(client, [1.55, 1.15, 0.003], [4.70, 3.85, 0.003], [0.48, 0.22, 0.20, 1.0])
    _visual_box(client, [1.25, 0.48, 0.30], [5.05, 5.05, 0.30], [0.20, 0.35, 0.56, 1.0])
    _visual_box(client, [1.16, 0.14, 0.48], [5.05, 5.48, 0.84], [0.17, 0.30, 0.49, 1.0])
    _visual_box(client, [0.14, 0.44, 0.34], [3.98, 5.05, 0.48], [0.18, 0.32, 0.52, 1.0])
    _visual_box(client, [0.14, 0.44, 0.34], [6.12, 5.05, 0.48], [0.18, 0.32, 0.52, 1.0])
    _visual_box(client, [0.67, 0.43, 0.045], [4.65, 3.75, 0.48], [0.48, 0.30, 0.17, 1.0])
    for x in (4.10, 5.20):
        for y in (3.42, 4.08):
            _visual_box(client, [0.035, 0.035, 0.22], [x, y, 0.23], desk_wood)

    # Floor lamp beside the sofa.
    lamp_base = client.createVisualShape(
        client.GEOM_CYLINDER, radius=0.22, length=0.05, rgbaColor=[0.22, 0.20, 0.18, 1.0]
    )
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=lamp_base, basePosition=[3.48, 5.85, 0.025])
    lamp_pole = client.createVisualShape(
        client.GEOM_CYLINDER, radius=0.025, length=1.35, rgbaColor=[0.18, 0.17, 0.15, 1.0]
    )
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=lamp_pole, basePosition=[3.48, 5.85, 0.72])
    lamp_shade = client.createVisualShape(
        client.GEOM_CYLINDER, radius=0.24, length=0.32, rgbaColor=[0.91, 0.78, 0.53, 1.0]
    )
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=lamp_shade, basePosition=[3.48, 5.85, 1.52])

    # Framed wall art and mirror on the left wall, clear of the bookshelf.
    for y, color in ((1.0, [0.25, 0.48, 0.55, 1.0]), (3.8, [0.69, 0.39, 0.21, 1.0])):
        _visual_box(client, [0.035, 0.55, 0.42], [-5.40, y, 1.60], trim)
        _visual_box(client, [0.012, 0.48, 0.35], [-5.36, y, 1.60], color)
    _visual_box(client, [0.035, 0.43, 0.62], [-5.40, -2.0, 1.58], trim)
    _visual_box(client, [0.012, 0.36, 0.55], [-5.36, -2.0, 1.58], [0.63, 0.79, 0.83, 0.75])

    # Plants decorate two corners, well outside the driving and manipulation area.
    _visual_plant(client, 5.90, 6.45)
    _visual_plant(client, -5.00, -4.35)
