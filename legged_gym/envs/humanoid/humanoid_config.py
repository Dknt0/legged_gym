from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class HumanoidRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15  # Policy frame stack number
        c_frame_stack = 3  # Critic frame stack number
        num_single_obs = 47  # 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73  # 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False  # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hhfc/urdf/hhfc-realfoot.urdf"

        name = "hhfc"
        foot_name = "ankle_r"
        knee_name = "knee"

        terminate_after_contacts_on = ["base_thorax"]
        penalize_contacts_on = ["base_thorax"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "plane"
        mesh_type = "trimesh"
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "Lleg_hip_p_joint": 0.165647,
            "Lleg_hip_r_joint": 0.0,
            "Lleg_hip_y_joint": 0.0,
            "Lleg_knee_joint": -0.529741,
            "Lleg_ankle_p_joint": -0.301101,
            "Lleg_ankle_r_joint": 0.0,
            "Rleg_hip_p_joint": 0.165647,
            "Rleg_hip_r_joint": 0.0,
            "Rleg_hip_y_joint": 0.0,
            "Rleg_knee_joint": -0.529741,
            "Rleg_ankle_p_joint": -0.301101,
            "Rleg_ankle_r_joint": 0.0,
            "Larm_shoulder_p_joint": 0.0,
            "Rarm_shoulder_p_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # # PD Drive parameters:
        # stiffness = {
        #     "hip_r": 140.0,
        #     "hip_p": 130.0,
        #     "hip_y": 80.0,
        #     "knee": 140.0,
        #     "ankle": 60,
        #     "shoulder_p": 100.0,
        # }
        # damping = {
        #     "hip_r": 2.8,
        #     "hip_p": 2.6,
        #     "hip_y": 1.6,
        #     "knee": 2.8,
        #     "ankle": 1.2,
        #     "shoulder_p": 2.0,
        # }
        # PD Drive parameters:
        stiffness = {
            "hip_r": 120.0,
            "hip_p": 120.0,
            "hip_y": 80.0,
            "knee": 120.0,
            "ankle": 30,
            "shoulder_p": 100.0,
        }
        damping = {
            "hip_r": 1.2,
            "hip_p": 1.2,
            "hip_y": 1.0,
            "knee": 1.2,
            "ankle": 1.0,
            "shoulder_p": 2.0,
        }
        random_pd = False
        random_p_factor = 0.2
        random_d_factor = 0.2

        # stiffness = {
        #     "hip_r": 150.0,
        #     "hip_p": 250.0,
        #     "hip_y": 150.0,
        #     "knee": 250.0,
        #     "ankle": 50,
        #     "shoulder_p": 100.0,
        # }
        # # Unitree 2.5 4.0 2.0
        # damping = {
        #     "hip_r": 1.0,
        #     "hip_p": 1.0,
        #     "hip_y": 1.0,
        #     "knee": 1.0,
        #     "ankle": 1.0,
        #     "shoulder_p": 2.0,
        # }
        # This should be devided by action_scale
        action_upper_clip = [
            3.1416,
            0.52359,
            1.2217,
            0.0,
            0.6108,
            0.08727,
            3.1416,
            0.12217,
            1.0472,
            0.0,
            0.6108,
            0.08727,
            # 'Larm_shoulder_p_joint': 0.,
            # 'Rarm_shoulder_p_joint': 0.,
        ]
        action_lower_clip = [
            -1.5708,
            -0.12217,
            -1.0472,
            -1.5708,
            -0.6108,
            -0.08727,
            -1.5708,
            -0.52359,
            -1.2217,
            -1.5708,
            -0.6108,
            -0.08727,
            # 'Larm_shoulder_p_joint': 0.,
            # 'Rarm_shoulder_p_joint': 0.,
        ]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5  # 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]

        randomize_com = True
        com_displacement_range = [[-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03]]

        randomize_gains = False
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]  # Factor

        randomize_torque = False
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = False
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = False
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = False
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]  # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = False
        randomize_lag_timesteps = False
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]

        add_dof_lag = False
        randomize_dof_lag_timesteps = False
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]

        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]

        add_imu_lag = False
        randomize_imu_lag_timesteps = False
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]

        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]

        push_robots = True
        push_interval_s = 4  # 2 # 4
        max_push_vel_xy = 0.2  # 1.0 # 0.2
        max_push_ang_vel = 0.4  # 1.5 # 0.4
        # push_interval_s = 2  # 2 # 4
        # max_push_vel_xy = 1.0  # 0.2
        # max_push_ang_vel = 1.5  # 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = (
            False  # True  # if true: compute ang vel command from heading error
        )

        class ranges:
            # lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            # ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            lin_vel_x = [-0.8, 0.8]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.87  # 0.87  # 0.89
        min_dist = 0.24  # 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        # target_joint_pos_scale = 0.17    # rad
        target_joint_pos_scale = 0.32  # 0.32    # rad
        target_feet_height = 0.09  # 0.075  # 0.06        # m
        cycle_time = 0.64  # 0.80  # 0.64  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True  # True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5  # 5
        max_contact_force = 800  # 700  # Forces above this value are penalized
        trajectory_type = "sin2"  # "sin", "sin2", "sin3"

        class scales:
            # reference motion tracking
            joint_pos = 5.6  # 3.6 # 5.6
            feet_clearance = 1.0  # 0.1 # 2.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 2.0  # 1.0
            foot_slip = -0.05
            feet_distance = 0.2  # 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01  # -0.01
            # vel tracking
            tracking_lin_vel = 3.6  # 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 1.0  # 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 2.0  # 2.0  # 1.
            base_height = 0.3  # 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002  # -4.0 # -1.0 # -0.2 # -0.002
            torques = -1e-5
            dof_vel = -5e-4  # -1e-3  # -5e-4
            dof_acc = -1e-7  # -5e-6 # -1e-6  # -1e-7
            collision = -1.0

            action_rate = -0.0  # -8.0 # 1.0 # -0.2 # -0.0
            ankle_torques = -5e-5  # -5e-5

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    class debug_log:
        debug_log = False  # False
        save_csv_file_name = (
            "/home/dknt/Project/rl/legged_gym/legged_gym/scripts/csv/0.csv"
        )

        q_keywords = [
            "joint_angle_00",
            "joint_angle_01",
            "joint_angle_02",
            "joint_angle_03",
            "joint_angle_04",
            "joint_angle_05",
            "joint_angle_10",
            "joint_angle_11",
            "joint_angle_12",
            "joint_angle_13",
            "joint_angle_14",
            "joint_angle_15",
        ]
        dq_keywords = [
            "joint_velo_00",
            "joint_velo_01",
            "joint_velo_02",
            "joint_velo_03",
            "joint_velo_04",
            "joint_velo_05",
            "joint_velo_10",
            "joint_velo_11",
            "joint_velo_12",
            "joint_velo_13",
            "joint_velo_14",
            "joint_velo_15",
        ]
        torque_keywords = [
            "joint_torque_00",
            "joint_torque_01",
            "joint_torque_02",
            "joint_torque_03",
            "joint_torque_04",
            "joint_torque_05",
            "joint_torque_10",
            "joint_torque_11",
            "joint_torque_12",
            "joint_torque_13",
            "joint_torque_14",
            "joint_torque_15",
        ]

        actions_keywords = [
            "input_last_action_tensor_0",
            "input_last_action_tensor_1",
            "input_last_action_tensor_2",
            "input_last_action_tensor_3",
            "input_last_action_tensor_4",
            "input_last_action_tensor_5",
            "input_last_action_tensor_6",
            "input_last_action_tensor_7",
            "input_last_action_tensor_8",
            "input_last_action_tensor_9",
            "input_last_action_tensor_10",
            "input_last_action_tensor_11",
        ]
        lin_vel_keywords = [
            "CoM_linear_velo_0",
            "CoM_linear_velo_1",
            "CoM_linear_velo_2",
        ]
        ang_vel_keywords = ["CoM_angle_velo_0", "CoM_angle_velo_1", "CoM_angle_velo_2"]
        proj_grav_keywords = [
            "proj_grav_GT_x",
            "proj_grav_GT_y",
            "proj_grav_GT_no_noise_z",
        ]
        clock_keywords = ["input_clock_0", "input_clock_1"]

        ref_trajectory = [
            "ref_left_hip_pitch",
            "ref_left_hip_roll",
            "ref_left_hip_yaw",
            "ref_left_knee",
            "ref_left_ankle_pitch",
            "ref_left_ankle_roll",
            "ref_right_hip_pitch",
            "ref_right_hip_roll",
            "ref_right_hip_yaw",
            "ref_right_knee",
            "ref_right_ankle_pitch",
            "ref_right_ankle_roll",
        ]

        max_count = 1000

        # log_vars = ['actions']
        log_vars = ["actions", "t", "torques", "dof_pos", "dof_vel", "base_ang_vel"]
        # , 'episode_time'
        log_period_s = 5


class HumanoidRobotCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = (
            100  # Please check for potential savings every `save_interval` iterations.
        )
        experiment_name = "humanoid"
        run_name = ""
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
