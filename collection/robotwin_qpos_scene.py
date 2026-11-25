"""Project-local fixed left-then-right scene and wrist-camera calibration."""
import importlib
import numpy as np
import sapien


def make_left_right_env(make_env, seed, collect, wrist_camera=None, head_camera=None,
                        stack_xy_jitter_m=0., fixed_block_yaw=False, layout=None,
                        failure_offset_xy_m=None):
    if layout is not None:
        fixed_block_yaw = layout.get('fixed_block_yaw', False)
    module = importlib.import_module('envs.stack_blocks_two')
    original = module.rand_pose
    calls = []

    def sample(**kwargs):
        # First actor is red (left), second is green (right). Both stay far
        # from x=0. These small pilot ranges are recorded, not official defaults.
        index = len(calls)
        assert index < 2, 'Unexpected actor rejection: review placement constraints'
        kwargs.update(xlim=[-.24, -.20] if index == 0 else [.20, .24],
                      ylim=[-.04, 0.0])
        if fixed_block_yaw:
            kwargs['rotate_rand'] = False
        pose = original(**kwargs)
        calls.append(pose)
        return pose

    module.rand_pose = sample
    try:
        env, config = make_env(seed, collect=collect, embodiment='piper')
    finally:
        module.rand_pose = original
    assert len(calls) == 2
    config['project2_layout'] = dict(red_x=[-.24, -.20], green_x=[.20, .24],
                                     y=[-.04, 0.0], order=['left', 'right'],
                                     yaw_range_rad=[0., 0.] if fixed_block_yaw else [-.75, .75],
                                     fixed_block_yaw=fixed_block_yaw)
    # Separate RNG: changing the target never changes the paired initial scene.
    assert 0. <= stack_xy_jitter_m <= .02
    offset = np.random.default_rng(np.random.SeedSequence([seed, 58321])).uniform(
        -stack_xy_jitter_m, stack_xy_jitter_m, size=2)
    if layout is not None:
        offset = np.asarray(layout.get('stack_offset_xy_m', [0., 0.]), dtype=float)
        failure_offset = np.asarray(layout.get('failure_offset_xy_m', [0., 0.]), dtype=float)
    else:
        failure_offset = np.zeros(2, dtype=float) if failure_offset_xy_m is None else np.asarray(failure_offset_xy_m, dtype=float)
    assert failure_offset.shape == (2,)
    target_xy = np.array([0., -.13]) + offset
    config['project2_layout'].update(stack_offset_xy_m=offset.tolist(),
                                     stack_target_xy_m=target_xy.tolist(),
                                     stack_xy_jitter_m=stack_xy_jitter_m if layout is None
                                     else layout.get('stack_xy_jitter_m', 0.),
                                     failure_offset_xy_m=failure_offset.tolist())
    env.block1_target_pose[:2] = target_xy.tolist()
    # The expert hardcodes the first placement target inside its method.
    # Adapt only that call locally; the green block still targets the red block.
    original_place = env.place_actor

    def place(actor, *args, **kwargs):
        if actor is env.block1:
            target = list(kwargs['target_pose'])
            target[:2] = target_xy.tolist()
            kwargs['target_pose'] = target
        elif actor is env.block2 and np.any(failure_offset):
            target = list(kwargs['target_pose'])
            target[:2] = (np.asarray(target[:2], dtype=float) + failure_offset).tolist()
            kwargs['target_pose'] = target
        return original_place(actor, *args, **kwargs)

    env.place_actor = place
    env.prohibited_area[-1] = (np.asarray(env.prohibited_area[-1]) +
                               np.tile(offset, 2)).tolist()
    # Use the URDF's actual camera link and its installed optical direction.
    # Retreat 4cm along the optical axis and offset outward 3.5cm to clear
    # the original camera housing. Keep the installed orientation unchanged.
    position = np.array([-.025, -.032, .035])
    matrix = np.eye(4)
    matrix[:3, 3] = position
    parent = 'camera'
    if wrist_camera is not None:
        matrix = np.asarray(wrist_camera['local_matrix'], dtype=float)
        parent = wrist_camera['parent']
    local = sapien.Pose(matrix)

    def update_wrist(_left_pose, _right_pose):
        for side in ('left', 'right'):
            link = getattr(env.robot, side + '_entity').find_link_by_name(parent)
            getattr(env.cameras, side + '_camera').entity.set_pose(link.get_pose() * local)

    env.cameras.update_wrist_camera = update_wrist
    near = .01 if wrist_camera is None else wrist_camera['near_m']
    fovy = 95. if wrist_camera is None else wrist_camera.get('fovy_deg')
    for side in ('left', 'right'):
        getattr(env.cameras, side + '_camera').set_near(near)
        if fovy is not None:
            getattr(env.cameras, side + '_camera').set_fovy(np.deg2rad(fovy))
    config['project2_wrist_camera'] = dict(parent=parent, local_matrix=matrix.tolist(),
                                          near_m=near, uses_object_truth=False)
    if fovy is not None:
        config['project2_wrist_camera']['fovy_deg'] = fovy
    # Historical episodes without a head override retain their original camera.
    if head_camera is None and wrist_camera is None:
        position = np.array([0., -.70, 1.70])
        forward = np.array([0., -.06, .83]) - position
        forward /= np.linalg.norm(forward)
        left = np.cross([0., 0., 1.], forward)
        left /= np.linalg.norm(left)
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack([forward, left, np.cross(forward, left)])
        matrix[:3, 3] = position
        head_camera = dict(world_matrix=matrix.tolist(), uses_object_truth=False)
    if head_camera is not None:
        camera = env.cameras.static_camera_list[env.cameras.head_camera_id]
        camera.entity.set_pose(sapien.Pose(np.asarray(head_camera['world_matrix'])))
        config['project2_head_camera'] = head_camera
    return env, config
