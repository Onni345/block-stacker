from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from pick_n_place.utils.xml_utils import replace_simhive_path
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from pathlib import Path
import click
import numpy as np
import time

DESC = """
stack two square blocks with franka arm
run: mjpython main_stack.py -s pick_place_stack.xml
"""

# ---------- constants ----------
STACK_SITE = "stack_target"
BLOCK_NAMES = ["box1", "box2"]
ARM_NJNT = 7                    # franka has 7 arm joints
CLEAR_Z = 0.3                   # high vertical clearance for safety
GRASP_Z_OFF = 0.002             # very light press into the block
GRIPPER_OPEN = 0.04             # open gripper value
GRIPPER_CLOSED = 0.0            # closed gripper value

# ---------- path planning helpers ----------
def stabilize_simulation(sim, steps=20, render=False):
    """run simulation steps to stabilize physics"""
    for _ in range(steps):
        sim.advance(render=render)

def plan(sim, start_q, target_pos, target_quat, T, dt):
    """generate a trajectory using inverse kinematics and min-jerk planning"""
    ik = qpos_from_site_pose(
        physics=sim, site_name="end_effector",
        target_pos=target_pos, target_quat=target_quat,
        inplace=False, regularization_strength=1.0)
    return generate_joint_space_min_jerk(start_q, ik.qpos[:ARM_NJNT], T, dt)

def get_block_dimensions(sim, body_id):
    """get the dimensions of a block from its geom size"""
    geom_id = sim.model.body_geomadr[body_id]
    return sim.model.geom_size[geom_id].copy()

def detect_contact(sim, box_bid, threshold=0.001):
    """
    detect if the box has made contact with a surface
    returns true if contact force exceeds threshold
    """
    # check contact forces for the block
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        # check if this block is involved in the contact
        if (contact.geom1 in sim.model.body_geomid[box_bid] or
            contact.geom2 in sim.model.body_geomid[box_bid]):
            # if there's significant contact force
            if np.linalg.norm(contact.force) > threshold:
                return True
    return False

def execute_ik_precise(sim, start_pos, end_pos, quat, T, dt, gripper_val, strict_xy=False):
    """
    execute a trajectory with precise ik with option to enforce strict xy alignment

    parameters:
    -----------
    sim : simscene
        simulation environment
    start_pos : np.array
        starting position
    end_pos : np.array
        ending position
    quat : np.array
        orientation quaternion
    t : float
        time horizon
    dt : float
        time step
    gripper_val : float
        gripper position value
    strict_xy : bool
        if true maintain exact xy values from end_pos throughout trajectory
    """
    # generate trajectory waypoints
    steps = int(T/dt)

    # current joint positions
    current_q = sim.data.ctrl[:ARM_NJNT].copy()

    # execute each waypoint with fresh ik
    for i in range(steps):
        t = i / (steps - 1)
        # for strict xy only interpolate z component
        if strict_xy:
            pos = np.array([
                end_pos[0],           # keep target x
                end_pos[1],           # keep target y
                start_pos[2] * (1-t) + end_pos[2] * t  # interpolate only z
            ])
        else:
            # regular interpolation of all components
            pos = start_pos * (1-t) + end_pos * t

        # compute ik for current waypoint
        ik = qpos_from_site_pose(
            physics=sim, site_name="end_effector",
            target_pos=pos, target_quat=quat,
            inplace=False, regularization_strength=1.0)

        # apply joint positions and gripper
        sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
        sim.data.ctrl[-1] = sim.data.ctrl[-2] = gripper_val
        sim.advance(render=True)

        # update current joint positions
        current_q = sim.data.ctrl[:ARM_NJNT].copy()

    return current_q

# ---------- cli ----------
@click.command(help=DESC)
@click.option("-s", "--sim_path", default="pick_place_stack.xml")
@click.option("-h", "--horizon", default=2, type=int, help="seconds per leg")
@click.option("-v", "--verbose", is_flag=True, help="print detailed info")
def main(sim_path, horizon, verbose):
    """main function for block stacking demonstration"""

    # --- load env ---
    sim_path = Path(__file__).parents[1] / "env" / sim_path
    sim_xml = replace_simhive_path(str(sim_path))
    print(f"loading {sim_xml}")
    sim = SimScene.get_sim(model_handle=sim_xml)

    # --- ids / baseline ---
    stack_sid = sim.model.site_name2id(STACK_SITE)
    box_bids = [sim.model.body_name2id(n) for n in BLOCK_NAMES]
    ARM_HOME = np.mean(sim.model.jnt_range[:ARM_NJNT], axis=-1)
    dt = sim.model.opt.timestep
    down_quat = np.array([0, 1, 0, 0])  # tool z-axis down

    # --- stacking state tracking ---
    stack_level = 0
    cur_box_idx = 0
    first_block_placed = False

    # run until manually stopped
    while True:
        print(f"\n===== starting placement of block {cur_box_idx+1} =====")

        # only reset simulation for the first block or when starting over
        if not first_block_placed or cur_box_idx == 0:
            sim.reset()
            stabilize_simulation(sim, steps=30)
            first_block_placed = False
        else:
            # for subsequent blocks just stabilize the simulation without resetting
            # to maintain the position of previously placed blocks
            stabilize_simulation(sim, steps=10)


        # get current block position and dimensions
        cur_bid = box_bids[cur_box_idx]
        box_pos = sim.data.xpos[cur_bid].copy()
        box_dims = get_block_dimensions(sim, cur_bid)

        # determine stack target position
        stack_pos_base = sim.model.site_pos[stack_sid].copy()

        # for first block place on stack target site
        # for subsequent blocks place on top of previous block
        if stack_level == 0:
            stack_pos = stack_pos_base.copy()
        else:
            # if we need to check the previous block's position (for stacking)
            if stack_level > 0:
                prev_bid = box_bids[cur_box_idx - 1]
                prev_pos = sim.data.xpos[prev_bid].copy()
                prev_dims = get_block_dimensions(sim, prev_bid)

                # use the actual position of previous block plus its height
                # for square blocks the height is the z dimension
                stack_pos = np.array([
                    stack_pos_base[0],
                    stack_pos_base[1],
                    prev_pos[2] + prev_dims[2] * 2  # top of previous block
                ])

                if verbose:
                    print(f"previous block actual position: {prev_pos}")
                    print(f"stacking on top at: {stack_pos}")

        if verbose:
            print(f"block position: {box_pos}")
            print(f"block dimensions: {box_dims}")
            print(f"stack target: {stack_pos}")

        # ---------- stage 1 move to safe position above block ----------
        print("moving to position above block")

        # first move arm to home position
        go_home = generate_joint_space_min_jerk(
            sim.data.ctrl[:ARM_NJNT],
            ARM_HOME,
            horizon/2,
            dt
        )

        for step in go_home:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # then move to position directly above the block
        # calculate position at the top center of the block
        precise_top_center = np.array([
            box_pos[0],
            box_pos[1],
            box_pos[2] + box_dims[2] + CLEAR_Z  # high above block
        ])

        # move to position above block
        move_to_high = plan(sim, sim.data.ctrl[:ARM_NJNT], precise_top_center, down_quat, horizon, dt)

        # execute trajectory
        for step in move_to_high:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # now verify we're perfectly aligned with the box center in xy
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        precise_align_pos = np.array([
            box_pos[0],  # exact box x
            box_pos[1],  # exact box y
            current_pos[2]  # keep current height
        ])

        # use precision alignment (may take extra time but ensures accuracy)
        current_q = execute_ik_precise(
            sim,
            current_pos,
            precise_align_pos,
            down_quat,
            horizon/2,  # short time for small adjustment
            dt,
            GRIPPER_OPEN
        )

        # ---------- stage 2 descend to grasp block ----------
        print("descending to grasp block with precise xy alignment")

        # current gripper position after alignment
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # position to grasp at top-center of block
        grasp_pos = np.array([
            box_pos[0],  # exact box x
            box_pos[1],  # exact box y
            box_pos[2] + box_dims[2] - GRASP_Z_OFF  # top of block with slight press
        ])

        # precise descent with strict xy alignment
        current_q = execute_ik_precise(
            sim,
            current_pos,
            grasp_pos,
            down_quat,
            horizon * 1.5,  # extra time for careful descent
            dt,
            GRIPPER_OPEN,
            strict_xy=True  # critical maintain exact xy during descent
        )

        # ---------- stage 3 grasp block ----------
        print("grasping block")

        # hold position and close gripper
        gripper_close_steps = int(horizon/dt)

        for i in range(gripper_close_steps):
            # gradually close gripper
            close_factor = i / gripper_close_steps
            grip_val = GRIPPER_OPEN * (1 - close_factor) + GRIPPER_CLOSED * close_factor

            # use the last joint positions from precise descent
            sim.data.ctrl[:ARM_NJNT] = current_q
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = grip_val
            sim.advance(render=True)

        # ---------- stage 4 lift block up ----------
        print("lifting block")

        # current gripper position after grasping
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # lift straight up with block maintaining xy alignment
        lift_pos = np.array([
            current_pos[0],  # maintain current x
            current_pos[1],  # maintain current y
            current_pos[2] + CLEAR_Z  # lift by clearance amount
        ])

        # precise lift with strict xy alignment
        current_q = execute_ik_precise(
            sim,
            current_pos,
            lift_pos,
            down_quat,
            horizon,
            dt,
            GRIPPER_CLOSED,
            strict_xy=True  # maintain xy alignment during lift
        )

        # ---------- stage 5 move to position above stack target ----------
        print("moving to position above stack target")

        # current position after lift
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # first move higher for safety
        high_pos = np.array([
            current_pos[0],
            current_pos[1],
            current_pos[2] + 0.1  # even higher
        ])

        high_move = plan(sim, current_q, high_pos, down_quat, horizon/2, dt)

        for step in high_move:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_CLOSED
            sim.advance(render=True)

        # move directly above stack target
        stack_high = np.array([
            stack_pos[0],  # exact stack target x
            stack_pos[1],  # exact stack target y
            high_pos[2]    # maintain high z for safety
        ])

        # execute move to above stack position
        move_to_stack = plan(sim, sim.data.ctrl[:ARM_NJNT], stack_high, down_quat, horizon, dt)

        for step in move_to_stack:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_CLOSED
            sim.advance(render=True)

        # verify precise xy alignment with stack target
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        precise_stack_align = np.array([
            stack_pos[0],  # exact stack x
            stack_pos[1],  # exact stack y
            current_pos[2]  # keep current height
        ])

        # fine alignment above stack target
        current_q = execute_ik_precise(
            sim,
            current_pos,
            precise_stack_align,
            down_quat,
            horizon/2,
            dt,
            GRIPPER_CLOSED
        )

        # ---------- stage 6 place block carefully ----------
        print("carefully placing block")

        # current position after alignment
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # calculate a position slightly above stack target for initial descent
        pre_place_pos = np.array([
            stack_pos[0],
            stack_pos[1],
            stack_pos[2] + 0.03  # small offset above final position
        ])

        # first descend to just above target with strict xy
        current_q = execute_ik_precise(
            sim,
            current_pos,
            pre_place_pos,
            down_quat,
            horizon * 1.5,
            dt,
            GRIPPER_CLOSED,
            strict_xy=True
        )

        # now do the final descent with contact detection
        print("final descent with contact detection")
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # get current block id
        cur_bid = box_bids[cur_box_idx]

        # calculate total steps for final descent
        final_descent_steps = int(horizon * 1.0 / dt)
        step_size = (pre_place_pos[2] - stack_pos[2]) / final_descent_steps

        # start with gripper closed
        grip_val = GRIPPER_CLOSED
        contact_detected = False
        release_started = False
        release_complete = False

        # execute final descent with contact detection for early release
        for i in range(final_descent_steps):
            if sim.advance(render=True) != 0:
                print("simulation error occurred")
                break

            # current position and target
            current_ee_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
            target_z = pre_place_pos[2] - (i+1) * step_size
            target_pos = np.array([stack_pos[0], stack_pos[1], target_z])

            # check for contact between block and target surface
            if not contact_detected:
                # adjusted threshold for square blocks
                contact_detected = detect_contact(sim, cur_bid, threshold=0.05)
                if contact_detected:
                    print("contact detected beginning immediate gripper release")

            # start opening gripper immediately upon contact
            if contact_detected and not release_complete:
                if not release_started:
                    release_started = True

                # very rapid gripper opening over 5 steps once contact is detected
                if release_started and grip_val > GRIPPER_OPEN:
                    grip_val = max(GRIPPER_OPEN, grip_val - (GRIPPER_CLOSED - GRIPPER_OPEN)/5)
                    if grip_val == GRIPPER_OPEN:
                        release_complete = True
                        print("gripper fully released upon contact")

            # perform ik for current descent position
            ik = qpos_from_site_pose(
                physics=sim, site_name="end_effector",
                target_pos=target_pos, target_quat=down_quat,
                inplace=False, regularization_strength=1.0)

            # apply controls - maintain xy alignment throughout
            sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = grip_val

            # if we've fully released move slightly up to avoid pushing the block
            if release_complete and i < final_descent_steps - 1:
                # slight upward adjustment after release
                release_pos = np.array([
                    stack_pos[0],
                    stack_pos[1],
                    current_ee_pos[2] + 0.005  # small upward adjustment
                ])

                ik = qpos_from_site_pose(
                    physics=sim, site_name="end_effector",
                    target_pos=release_pos, target_quat=down_quat,
                    inplace=False,
                    regularization_strength=1.0)

                sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
                break  # exit the descent loop once we've released and adjusted

        # update current joint positions
        current_q = sim.data.ctrl[:ARM_NJNT].copy()

        # if we didn't detect contact ensure gripper is fully open anyway
        if not contact_detected:
            print("no contact detected during placement releasing gripper")
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # ---------- stage 7 release block (if not already released during placement) ----------
        print("ensuring block is fully released")

        # quick open gripper if not already fully open
        if not (release_complete and contact_detected):
            quick_open_steps = int(horizon/dt * 0.2)  # very fast opening
            for i in range(quick_open_steps):
                open_factor = i / quick_open_steps
                grip_val = GRIPPER_CLOSED * (1 - open_factor) + GRIPPER_OPEN * open_factor

                sim.data.ctrl[:ARM_NJNT] = current_q
                sim.data.ctrl[-1] = sim.data.ctrl[-2] = grip_val
                sim.advance(render=True)

        # fully open gripper
        sim.data.ctrl[:ARM_NJNT] = current_q
        sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
        sim.advance(render=True)

        # ---------- stage 8 retreat safely ----------
        print("retreating from stack")

        # add a small delay to let the block settle completely before retreating
        for _ in range(int(horizon/dt * 0.2)):
            sim.data.ctrl[:ARM_NJNT] = current_q
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # current position after release
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # first do a quick but small vertical movement to clear the block
        initial_retreat_pos = np.array([
            current_pos[0],  # keep x
            current_pos[1],  # keep y
            current_pos[2] + 0.05  # small initial retreat
        ])

        # quick initial retreat
        for _ in range(int(horizon/dt * 0.2)):
            # compute ik for initial retreat
            ik = qpos_from_site_pose(
                physics=sim, site_name="end_effector",
                target_pos=initial_retreat_pos,
                target_quat=down_quat,
                inplace=False,
                regularization_strength=1.0)

            # apply controls
            sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # update current position and joint angles
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        current_q = sim.data.ctrl[:ARM_NJNT].copy()

        # now do the full retreat
        full_retreat_pos = np.array([
            current_pos[0],  # keep x
            current_pos[1],  # keep y
            current_pos[2] + CLEAR_Z - 0.05  # full retreat (accounting for initial)
        ])

        # execute full vertical retreat
        retreat = plan(sim, current_q, full_retreat_pos, down_quat, horizon, dt)

        for step in retreat:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # update current joint angles
        current_q = sim.data.ctrl[:ARM_NJNT].copy()

        # then move to home position
        go_home = generate_joint_space_min_jerk(
            current_q,
            ARM_HOME,
            horizon,
            dt
        )

        for step in go_home:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)

        # ---------- evaluate placement success ----------
        # let physics settle more extensively to ensure stability
        stabilize_simulation(sim, steps=80, render=True)

        # measure placement error
        cur_bid = box_bids[cur_box_idx]
        actual_pos = sim.data.xpos[cur_bid]
        err_xy = np.linalg.norm(actual_pos[:2] - stack_pos[:2])
        err_z = abs(actual_pos[2] - stack_pos[2])
        err_total = err_xy + err_z

        print(f"\nplacement results for block {cur_box_idx+1}:")
        print(f"  target position: {stack_pos}")
        print(f"  actual position: {actual_pos}")
        print(f"  error: xy={err_xy:.3f}m z={err_z:.3f}m total={err_total:.3f}m")

        # check if placement was successful
        success = err_total < 0.15  # 15cm tolerance

        if success:
            print(f"successfully placed block {cur_box_idx+1}")

            # mark first block as placed if this is the first one
            if cur_box_idx == 0:
                first_block_placed = True

            # advance to next block
            cur_box_idx += 1
            stack_level += 1

            # reset if all blocks stacked
            if cur_box_idx >= len(box_bids):
                print("tower complete resetting")
                cur_box_idx = 0
                stack_level = 0
                first_block_placed = False
                time.sleep(1)  # pause to see completed stack
        else:
            print(f"failed to place block {cur_box_idx+1} correctly")

            # if failed during the second block don't reset first block
            if cur_box_idx == 1:
                print("maintaining first block position and retrying second block placement")

        # small pause between attempts
        time.sleep(0.5)

if __name__ == "__main__":
    main()