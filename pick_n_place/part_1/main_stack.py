from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from pick_n_place.utils.xml_utils import replace_simhive_path
from pathlib import Path
import click
import numpy as np
import time

DESC = """
Stack two square blocks with Franka arm.
Run:  mjpython main_stack.py -s pick_place_stack.xml
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
def stabilize_simulation(sim, steps=20, render=False):
    """Run simulation steps to stabilize physics"""
    for _ in range(steps):
        sim.advance(render=render)

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

def detect_contact(sim, box_bid, threshold=0.001):
    """
    Detect if the box has made contact with a surface
    Returns True if contact force exceeds threshold
    """
    # Check contact forces for the block
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        # Check if this block is involved in the contact
        if (contact.geom1 in sim.model.body_geomid[box_bid] or 
            contact.geom2 in sim.model.body_geomid[box_bid]):
            # If there's significant contact force
            if np.linalg.norm(contact.force) > threshold:
                return True
    return False

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
@click.option("-s", "--sim_path", default="pick_place_stack.xml")
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
            # If we need to check the previous block's position (for stacking)
            if stack_level > 0:
                prev_bid = box_bids[cur_box_idx - 1]
                prev_pos = sim.data.xpos[prev_bid].copy()
                prev_dims = get_block_dimensions(sim, prev_bid)
                
                # Use the ACTUAL position of previous block plus its height
                # For square blocks, the height is the Z dimension
                stack_pos = np.array([
                    stack_pos_base[0],
                    stack_pos_base[1],
                    prev_pos[2] + prev_dims[2] * 2  # Top of previous block
                ])
                
                if verbose:
                    print(f"Previous block actual position: {prev_pos}")
                    print(f"Stacking on top at: {stack_pos}")
        
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
        
        # Calculate a position slightly above stack target for initial descent
        pre_place_pos = np.array([
            stack_pos[0],
            stack_pos[1],
            stack_pos[2] + 0.03  # Small offset above final position
        ])
        
        # First descend to just above target with strict XY
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
        
        # Now do the final descent with contact detection
        print("Final descent with contact detection...")
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        
        # Get current block ID
        cur_bid = box_bids[cur_box_idx]
        
        # Calculate total steps for final descent
        final_descent_steps = int(horizon * 1.0 / dt)
        step_size = (pre_place_pos[2] - stack_pos[2]) / final_descent_steps
        
        # Start with gripper closed
        grip_val = GRIPPER_CLOSED
        contact_detected = False
        release_started = False
        release_complete = False
        
        # Execute final descent with contact detection for early release
        for i in range(final_descent_steps):
            if sim.advance(render=True) != 0:
                print("Simulation error occurred")
                break
            
            # Current position and target
            current_ee_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
            target_z = pre_place_pos[2] - (i+1) * step_size
            target_pos = np.array([stack_pos[0], stack_pos[1], target_z])
            
            # Check for contact between block and target surface
            if not contact_detected:
                # Adjusted threshold for square blocks
                contact_detected = detect_contact(sim, cur_bid, threshold=0.05)
                if contact_detected:
                    print("⚠️ Contact detected! Beginning immediate gripper release...")
            
            # Start opening gripper immediately upon contact
            if contact_detected and not release_complete:
                if not release_started:
                    release_started = True
                    
                # Very rapid gripper opening over 5 steps once contact is detected
                if release_started and grip_val > GRIPPER_OPEN:
                    grip_val = max(GRIPPER_OPEN, grip_val - (GRIPPER_CLOSED - GRIPPER_OPEN)/5)
                    if grip_val == GRIPPER_OPEN:
                        release_complete = True
                        print("🔓 Gripper fully released upon contact")
            
            # Perform IK for current descent position
            ik = qpos_from_site_pose(
                physics=sim, site_name="end_effector",
                target_pos=target_pos, target_quat=down_quat,
                inplace=False, regularization_strength=1.0)
            
            # Apply controls - maintain XY alignment throughout
            sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = grip_val
            
            # If we've fully released, move slightly up to avoid pushing the block
            if release_complete and i < final_descent_steps - 1:
                # Slight upward adjustment after release
                release_pos = np.array([
                    stack_pos[0], 
                    stack_pos[1], 
                    current_ee_pos[2] + 0.005  # Small upward adjustment
                ])
                
                ik = qpos_from_site_pose(
                    physics=sim, site_name="end_effector",
                    target_pos=release_pos, target_quat=down_quat,
                    inplace=False, regularization_strength=1.0)
                
                sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
                break  # Exit the descent loop once we've released and adjusted
        
        # Update current joint positions
        current_q = sim.data.ctrl[:ARM_NJNT].copy()
        
        # If we didn't detect contact, ensure gripper is fully open anyway
        if not contact_detected:
            print("No contact detected during placement, releasing gripper...")
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)
        
        # ---------- STAGE 7: RELEASE BLOCK (if not already released during placement) ----------
        print("Ensuring block is fully released...")
        
        # Quick open gripper if not already fully open
        if not (release_complete and contact_detected):
            quick_open_steps = int(horizon/dt * 0.2)  # Very fast opening
            for i in range(quick_open_steps):
                open_factor = i / quick_open_steps
                grip_val = GRIPPER_CLOSED * (1 - open_factor) + GRIPPER_OPEN * open_factor
                
                sim.data.ctrl[:ARM_NJNT] = current_q
                sim.data.ctrl[-1] = sim.data.ctrl[-2] = grip_val
                sim.advance(render=True)
        
        # Fully open gripper
        sim.data.ctrl[:ARM_NJNT] = current_q
        sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
        sim.advance(render=True)
        
        # ---------- STAGE 8: RETREAT SAFELY ----------
        print("Retreating from stack...")
        
        # Add a small delay to let the block settle completely before retreating
        for _ in range(int(horizon/dt * 0.2)):
            sim.data.ctrl[:ARM_NJNT] = current_q
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)
        
        # Current position after release
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        
        # First do a quick but small vertical movement to clear the block
        initial_retreat_pos = np.array([
            current_pos[0],  # Keep X
            current_pos[1],  # Keep Y
            current_pos[2] + 0.05  # Small initial retreat
        ])
        
        # Quick initial retreat 
        for _ in range(int(horizon/dt * 0.2)):
            # Compute IK for initial retreat
            ik = qpos_from_site_pose(
                physics=sim, site_name="end_effector",
                target_pos=initial_retreat_pos, 
                target_quat=down_quat,
                inplace=False, 
                regularization_strength=1.0)
            
            # Apply controls
            sim.data.ctrl[:ARM_NJNT] = ik.qpos[:ARM_NJNT]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)
        
        # Update current position and joint angles
        current_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        current_q = sim.data.ctrl[:ARM_NJNT].copy()
        
        # Now do the full retreat
        full_retreat_pos = np.array([
            current_pos[0],  # Keep X
            current_pos[1],  # Keep Y
            current_pos[2] + CLEAR_Z - 0.05  # Full retreat (accounting for initial)
        ])
        
        # Execute full vertical retreat
        retreat = plan(sim, current_q, full_retreat_pos, down_quat, horizon, dt)
        
        for step in retreat:
            sim.data.ctrl[:ARM_NJNT] = step["position"]
            sim.data.ctrl[-1] = sim.data.ctrl[-2] = GRIPPER_OPEN
            sim.advance(render=True)
            
        # Update current joint angles
        current_q = sim.data.ctrl[:ARM_NJNT].copy()
        
        # Then move to home position
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
            print(f"❌ Failed to place block {cur_box_idx+1} correctly")
            
            # If failed during the second block, don't reset first block
            if cur_box_idx == 1:
                print("Maintaining first block position and retrying second block placement.")
        
        # Small pause between attempts
        time.sleep(0.5)

if __name__ == "__main__":
    main()