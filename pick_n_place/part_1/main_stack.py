# main_stack.py – completely rewritten block-stacking demo
from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from pick_n_place.utils.xml_utils import replace_simhive_path
from pathlib import Path
import click
import numpy as np
import time

DESC = """
Stack two blocks with Franka arm.
Run:  mjpython main_stack.py -s pick_place_stack_no_bins.xml
"""

# ---------- constants ----------
STACK_SITE = "stack_target"
BLOCK_NAMES = ["box1", "box2"]
ARM_NJNT = 7                   # Franka has 7 arm joints
CLEAR_Z = 0.3                  # High vertical clearance for safety
GRASP_Z_OFF = 0.002            # Very light press into the block
GRIPPER_OPEN = 0.04            # Open gripper value
GRIPPER_CLOSED = 0.0           # Closed gripper value

# ---------- Path Planning Helpers ----------
def plan(sim, start_q, target_pos, target_quat, T, dt):
    """Generate a trajectory using inverse kinematics and min-jerk planning"""
    ik = qpos_from_site_pose(
        physics=sim, site_name="end_effector",
        target_pos=target_pos, target_quat=target_quat,
        inplace=False, regularization_strength=1.0)
    return generate_joint_space_min_jerk(start_q, ik.qpos[:ARM_NJNT], T, dt)

def get_block_dimensions(sim, body_id):
    """Get the dimensions of a block from its geom size"""
    geom_id = sim.model.body_geomadr[body_id]
    return sim.model.geom_size[geom_id].copy()

def stabilize_simulation(sim, steps=20, render=False):
    """Run simulation steps to stabilize physics"""
    for _ in range(steps):
        sim.advance(render=render)

def execute_ik_precise(sim, start_pos, end_pos, quat, T, dt, gripper_val, strict_xy=False):
    """
    Execute a trajectory with precise IK, with option to enforce strict XY alignment
    
    Parameters:
    -----------
    sim : SimScene
        Simulation environment
    start_pos : np.array
        Starting position
    end_pos : np.array
        Ending position
    quat : np.array
        Orientation quaternion
    T : float
        Time horizon
    dt : float
        Time step
    gripper_val : float
        Gripper position value
    strict_xy : bool
        If True, maintain exact XY values from end_pos throughout trajectory
    """
    # Generate trajectory waypoints
    steps = int(T/dt)
    positions = []
    
    for i in range(steps):
        t = i / (steps - 1)
        # For strict XY, only interpolate Z component
        if strict_xy:
            pos = np.array([
                end_pos[0],             # Keep target X
                end_pos[1],             # Keep target Y
                start_pos[2] * (1-t) + end_pos[2] * t  # Interpolate only Z
            ])
        else:
            # Regular interpolation of all components
            pos = start_pos * (1-t) + end_pos * t
        positions.append(pos)
    
    # Current joint positions
    current_q = sim.data.ctrl[:ARM_NJNT].copy()
    
    # Execute each waypoint with fresh IK
    for pos in positions:
        # Compute IK for current waypoint
        ik = qpos_from_site_pose(
            physics=sim, site_name="end_effector",
            target_pos=pos, target_quat=quat,
            inplace=False, regularization_strength=1.0)
        
        # Apply joint positions and gripper
        sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
        sim.data.ctrl[-1] = sim.data.ctrl[-2] = gripper_val
        sim.advance(render=True)
        
        # Update current joint positions
        current_q = sim.data.ctrl[:ARM_NJNT].copy()
    
    return current_q

# ---------- CLI ----------
@click.command(help=DESC)
@click.option("-s", "--sim_path", default="pick_place_stack_no_bins.xml")
@click.option("-h", "--horizon", default=2, type=int, help="seconds per leg")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed info")
def main(sim_path, horizon, verbose):
    """Main function for block stacking demonstration"""
    
    # --- load env ---
    sim_path = Path(__file__).parents[1] / "env" / sim_path
    sim_xml = replace_simhive_path(str(sim_path))
    print(f"Loading {sim_xml}")
    sim = SimScene.get_sim(model_handle=sim_xml)

    # --- IDs / baseline ---
    stack_sid = sim.model.site_name2id(STACK_SITE)
    box_bids = [sim.model.body_name2id(n) for n in BLOCK_NAMES]
    ARM_HOME = np.mean(sim.model.jnt_range[:ARM_NJNT], axis=-1)
    dt = sim.model.opt.timestep
    down_quat = np.array([0, 1, 0, 0])  # tool Z-axis down

            # --- stacking state tracking ---
    stack_level = 0
    cur_box_idx = 0
    first_block_placed = False
    
    # Run until manually stopped
    while True:
        print(f"\n===== Starting placement of block {cur_box_idx+1} =====")
        
        # Only reset simulation for the first block or when starting over
        if not first_block_placed or cur_box_idx == 0:
            sim.reset()
            stabilize_simulation(sim, steps=30)
            first_block_placed = False
        else:
            # For subsequent blocks, just stabilize the simulation without resetting
            # to maintain the position of previously placed blocks
            stabilize_simulation(sim, steps=10)
        
        
        # Get current block position and dimensions
        cur_bid = box_bids[cur_box_idx]
        box_pos = sim.data.xpos[cur_bid].copy()
        box_dims = get_block_dimensions(sim, cur_bid)
        
        # Determine stack target position
        stack_pos_base = sim.model.site_pos[stack_sid].copy()
        
        # For first block, place on stack target site
        # For subsequent blocks, place on top of previous block
        if stack_level == 0:
            stack_pos = stack_pos_base.copy()
        else:
            # Get position of previous block
            prev_bid = box_bids[cur_box_idx - 1]
            prev_pos = sim.data.xpos[prev_bid].copy()
            prev_dims = get_block_dimensions(sim, prev_bid)
            
            # Stack XY from target, Z from previous block height
            stack_pos = np.array([
                stack_pos_base[0],
                stack_pos_base[1],
                prev_pos[2] + prev_dims[2] * 2  # Top of previous block
            ])
        
        if verbose:
            print(f"Block position: {box_pos}")
            print(f"Block dimensions: {box_dims}")
            print(f"Stack target: {stack_pos}")
        
        # ---------- STAGE 1: MOVE TO SAFE POSITION ABOVE BLOCK ----------
        print("Moving to position above block...")
        
        # First move arm to home position
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
            
        # Then move to position directly above the block
        # Calculate position at the top center of the block
        precise_top_center = np.array([
            box_pos[0],
            box_pos[1],
            box_pos[2] + box_dims[2] + CLEAR_Z  # High above block
        ])
        
        # Move to position above block
        move_to_high = plan(sim, sim.data.ctrl[:ARM_NJNT], precise_top_center, down_quat, horizon, dt)
        
        # Execute trajectory
        for step in move_to_high:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)
            
        # Now verify we're perfectly aligned with the box center in XY
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        precise_align_pos = np.array([
            box_pos[0],  # Exact box X
            box_pos[1],  # Exact box Y
            current_pos[2]  # Keep current height
        ])
        
        # Use precision alignment (may take extra time but ensures accuracy)
        current_q = execute_ik_precise(
            sim, 
            current_pos, 
            precise_align_pos, 
            down_quat, 
            horizon/2,  # Short time for small adjustment
            dt, 
            GRIPPER_OPEN
        )
        
        # ---------- STAGE 2: DESCEND TO GRASP BLOCK ----------
        print("Descending to grasp block with precise XY alignment...")
        
        # Current gripper position after alignment
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        
        # Position to grasp at top-center of block
        grasp_pos = np.array([
            box_pos[0],  # Exact box X
            box_pos[1],  # Exact box Y
            box_pos[2] + box_dims[2] - GRASP_Z_OFF  # Top of block with slight press
        ])
        
        # Precise descent with strict XY alignment
        current_q = execute_ik_precise(
            sim, 
            current_pos, 
            grasp_pos, 
            down_quat, 
            horizon * 1.5,  # Extra time for careful descent
            dt, 
            GRIPPER_OPEN,
            strict_xy=True  # Critical: maintain exact XY during descent
        )
        
        # ---------- STAGE 3: GRASP BLOCK ----------
        print("Grasping block...")
        
        # Hold position and close gripper
        gripper_close_steps = int(horizon/dt)
        
        for i in range(gripper_close_steps):
            # Gradually close gripper
            close_factor = i / gripper_close_steps
            grip_val = GRIPPER_OPEN * (1 - close_factor) + GRIPPER_CLOSED * close_factor
            
            # Use the last joint positions from precise descent
            sim.data.ctrl[:ARM_NJNT] = current_q
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = grip_val
            sim.advance(render=True)
        
        # ---------- STAGE 4: LIFT BLOCK UP ----------
        print("Lifting block...")
        
        # Current gripper position after grasping
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        
        # Lift straight up with block, maintaining XY alignment
        lift_pos = np.array([
            current_pos[0],  # Maintain current X
            current_pos[1],  # Maintain current Y
            current_pos[2] + CLEAR_Z  # Lift by clearance amount
        ])
        
        # Precise lift with strict XY alignment
        current_q = execute_ik_precise(
            sim, 
            current_pos, 
            lift_pos, 
            down_quat, 
            horizon, 
            dt, 
            GRIPPER_CLOSED,
            strict_xy=True  # Maintain XY alignment during lift
        )
        
        # ---------- STAGE 5: MOVE TO POSITION ABOVE STACK TARGET ----------
        print("Moving to position above stack target...")
        
        # Current position after lift
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        
        # First move higher for safety
        high_pos = np.array([
            current_pos[0],
            current_pos[1],
            current_pos[2] + 0.1  # Even higher
        ])
        
        high_move = plan(sim, current_q, high_pos, down_quat, horizon/2, dt)
        
        for step in high_move:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_CLOSED
            sim.advance(render=True)
        
        # Move directly above stack target
        stack_high = np.array([
            stack_pos[0],  # Exact stack target X
            stack_pos[1],  # Exact stack target Y
            high_pos[2]    # Maintain high Z for safety
        ])
        
        # Execute move to above stack position
        move_to_stack = plan(sim, sim.data.ctrl[:ARM_NJNT], stack_high, down_quat, horizon, dt)
        
        for step in move_to_stack:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_CLOSED
            sim.advance(render=True)
        
        # Verify precise XY alignment with stack target
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        precise_stack_align = np.array([
            stack_pos[0],  # Exact stack X
            stack_pos[1],  # Exact stack Y
            current_pos[2]  # Keep current height
        ])
        
        # Fine alignment above stack target
        current_q = execute_ik_precise(
            sim, 
            current_pos, 
            precise_stack_align, 
            down_quat, 
            horizon/2, 
            dt, 
            GRIPPER_CLOSED
        )
        
        # ---------- STAGE 6: PLACE BLOCK CAREFULLY ----------
        print("Carefully placing block...")

        # Current position after alignment
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()

        # Slow descent to place block with precise XY alignment
        current_q = execute_ik_precise(
            sim, 
            current_pos,
            stack_pos,
            down_quat, 
            horizon * 2.0,  # Extra slow for precision
            dt,
            GRIPPER_CLOSED,
            strict_xy=True  # Critical: maintain exact XY during descent
        )

        # Hold position briefly to stabilize (reduced hold time)
        hold_steps = int(horizon/dt * 0.05)  # Reduced from 0.2 to 0.05
        for _ in range(hold_steps):
            sim.data.ctrl[:ARM_NJNT] = current_q
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_CLOSED
            sim.advance(render=True)

        # ---------- STAGE 7: RELEASE BLOCK ----------
        print("Releasing block...")

        # Immediately open gripper completely
        sim.data.ctrl[:ARM_NJNT] = current_q
        sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
        sim.advance(render=True)

        # Let the block stabilize before moving (critical for stacked blocks)
        for _ in range(20):  # Added stabilization steps
            sim.data.ctrl[:ARM_NJNT] = current_q
            sim.advance(render=True)

        # ---------- STAGE 8: RETREAT SAFELY ----------
        print("Retreating from stack...")

        # Immediate vertical retreat to avoid collision
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        retreat_height = current_pos[2] + 0.15  # Immediate upward jump

        # Fast vertical retreat with strict XY
        current_q = execute_ik_precise(
            sim, 
            current_pos,
            [current_pos[0], current_pos[1], retreat_height],
            down_quat,
            horizon * 0.5,  # Faster retreat
            dt,
            GRIPPER_OPEN,
            strict_xy=True
        )

        # Continue to home position
        go_home = generate_joint_space_min_jerk(
            current_q,
            ARM_HOME,
            horizon * 0.5,  # Faster return
            dt
        )
        for step in go_home:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)
        
        # ---------- EVALUATE PLACEMENT SUCCESS ----------
        # Let physics settle more extensively to ensure stability
        stabilize_simulation(sim, steps=80, render=True)
        
        # Measure placement error
        cur_bid = box_bids[cur_box_idx]
        actual_pos = sim.data.xpos[cur_bid]
        err_xy = np.linalg.norm(actual_pos[:2] - stack_pos[:2])
        err_z = abs(actual_pos[2] - stack_pos[2])
        err_total = err_xy + err_z
        
        print(f"\nPlacement results for block {cur_box_idx+1}:")
        print(f"  Target position: {stack_pos}")
        print(f"  Actual position: {actual_pos}")
        print(f"  Error: XY={err_xy:.3f}m, Z={err_z:.3f}m, Total={err_total:.3f}m")
        
        # Check if placement was successful
        success = err_total < 0.15  # 5cm tolerance
        
        if success:
            print(f"✅ Successfully placed block {cur_box_idx+1}")
            
            # Mark first block as placed if this is the first one
            if cur_box_idx == 0:
                first_block_placed = True
                
            # Advance to next block
            cur_box_idx += 1
            stack_level += 1
            
            # Reset if all blocks stacked
            if cur_box_idx >= len(box_bids):
                print("🏆 Tower complete! Resetting...")
                cur_box_idx = 0
                stack_level = 0
                first_block_placed = False
                time.sleep(1)  # Pause to see completed stack
        else:
            print(f"❌ Failed to place block {cur_box_idx+1} correctly. Retrying.")
            # Don't advance, retry same block
            
            # If failed during the second block, don't reset first block
            if cur_box_idx == 1:
                print("Maintaining first block position and retrying second block placement.")
        
        # Small pause between attempts
        time.sleep(0.5)

if __name__ == "__main__":
    main()