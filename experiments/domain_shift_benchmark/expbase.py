dict(
    env_type='kitchen_shift',
    env_name='kitchen-v1',
    num_envs=1,
    env=dict(
        use_dummy_env=False,
        reset_on='either',
        zero_reward_on_success=True,
        elementwise_task_goal=False,
        # obs_keys=('robot_qp',),
        # obs_keys=('robot_qp', 'robot_qv'),
        obs_keys=('robot_qp', 'robot_qv', 'ee_qp', 'ee_forces'),
        # render_size=(128, 128),
        render_size=(256, 256),
        frame_size=(224, 224),
        kitchen_shift_params=dict(
            camera_id=6,
            ctrl_mode='absvel',
            # compensate_gravity=True,
            with_obs_ee=True,
            with_obs_forces=True,
            rot_use_euler=True,
            robot='franka2',
            noise_ratio=0.1,
            # object_pos_noise_amp=0.1,
            # object_vel_noise_amp=0.1,
            # robot_obs_noise_amp=0.1,
            #
            init_random_steps_set=[
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                75,
                80,
            ],
            init_perturb_robot_ratio=0.04,
            init_perturb_object_ratio=0.04,
            rng_type='generator',
        ),
        #
        # use_pixels=False,
        use_pixels=True,
        # grayscale=True,
        grayscale=False,
        demo_dir='./tmp_kitchenenv_demos/',
        demo_tag='1003c6r256_absvel',  # demo process env params should match ones used
        demo_type='full',
        # demo_type='singleobj',
        #
        # filter_demos=['friday_t-microwave,kettle,slide,hinge'],
        filter_demos=['friday_t-microwave,kettle,switch,slide'],
        # filter_demos=['friday_t-microwave', 'postcorl_t-microwave'],
        # filter_demos=['friday_t-kettle', 'postcorl_t-kettle'],
        # num_demos_per_task=25,
        task_goals=[
            # ['kettle', 'switch', 'hinge', 'slide'],
            # ['topknob', 'bottomknob', 'hinge', 'slide'],
            # ['microwave', 'bottomknob', 'hinge', 'slide'],
            # ['microwave', 'kettle', 'slide', 'hinge'],
            ['microwave', 'kettle', 'switch', 'slide'],
            # ['kettle', 'bottomknob', 'hinge', 'slide'],
            # ['kettle'],
            # ['microwave'],
        ],
    ),
    # num_eval_envs=1,
    num_eval_envs=10,
    eval_env=dict(
        reset_on='timeout',
        # max_episode_steps=70,
        # max_episode_steps=140,
        max_episode_steps=500,
        worldgen_wrapper=True,
        with_task_goal_id=True,  # for demo playback
        #
        # separate_render=True,  # if state vector inputs
    ),
    runner=dict(
        cls='train_eval_offline',
        num_iterations=100000,
        opt_steps=100,
        batch_size_opt=128,
        eval_interval=5000,
        ckpt_interval=25000,
        eval_func_params=dict(
            s0=dict(
                cls='eval_kitchenenv_generalize_single_object',
                # render_key='render',  # if state vector inputs
                num_evals_per_task=2,  # total also multiplied by num_eval_envs in parallel
                save_video_interval=5,
                deterministic_policy=False,
                # deterministic_policy=True,
                eval_related=False,
                eval_unrelated=False,
                # eval_unchanged=False,
                fix_seed=False,
            ),
            d0=dict(
                sub_eval_interval=100000,
                cls='eval_kitchenenv_domain_shift',
                # render_key='render',  # if state vector inputs
                num_evals_per_task=2,  # total also multiplied by num_eval_envs in parallel
                save_video_interval=5,
                deterministic_policy=False,
                fix_seed=False,
                domain_params=dict(
                    train=[[]],
                    change_object=(
                        [[('change_microwave', i)] for i in [1, 2, 3]]
                        + [[('change_kettle', i)] for i in [1, 2, 3, 4]]
                    ),
                    change_object_layout=(
                        [
                            [('change_objects_layout', 'microwave', i)]
                            for i in ['closer', 'closer_angled']
                        ]
                        + [
                            [('change_objects_layout', 'kettle', i)]
                            for i in [
                                'bot_left_angled',
                                'top_right',
                                'bot_right',
                                'bot_right_angled',
                            ]
                        ]
                        + [
                            [('change_objects_layout', 'slide', 'right_raised')],
                            [
                                ('change_objects_layout', 'hinge', 'left_lowered'),
                                ('change_objects_layout', 'slide', 'right_lowered'),
                                ('change_objects_layout', 'ovenhood', 'right_raised'),
                            ],
                        ]
                    ),
                    change_camera=(
                        [
                            [('change_camera', 2)],
                            [('change_camera', 7)],
                        ]
                    ),
                    change_lighting=(
                        [
                            [('change_lighting', i)]
                            for i in ['cast_left', 'cast_right', 'brighter', 'darker']
                        ]
                    ),
                    change_texture=(
                        [
                            [('change_hinge_texture', i)]
                            for i in [
                                'wood1',
                                # 'wood2',
                                'metal1',
                                # 'metal2',
                                'marble1',
                                # 'tile1',
                            ]
                        ]
                        + [
                            [('change_slide_texture', i)]
                            for i in [
                                # 'wood1',
                                'wood2',
                                # 'metal1',
                                'metal2',
                                # 'marble1',
                                'tile1',
                            ]
                        ]
                        + [
                            [('change_floor_texture', i)]
                            for i in [
                                # 'white_marble_tile',
                                # 'marble1',
                                'tile1',
                                'wood1',
                                # 'wood2',
                            ]
                        ]
                        + [
                            [('change_counter_texture', i)]
                            for i in [
                                # 'white_marble_tile2',
                                # 'tile1',
                                # 'wood1',
                                'wood2',
                            ]
                        ]
                    ),
                    change_noise=([[('change_noise_ratio', i)] for i in [0.5, 1.0, 10.0]]),
                    change_robot_init=(
                        [
                            [('change_robot_init_qpos', i)]
                            for i in [
                                [-0.4, -1.73, 1.76, -1.85, 0.15, 0.7, 1.7, 0.04, 0.04],
                                [-1.18, -1.76, 1.43, -1.57, -0.1, 0.88, 2.55, 0.0, -0.0],
                                [-1.62, -1.76, 0.6, -1.71, 0.36, 0.36, 1.7, 0.0, -0.0],
                                [-1.15, -1.76, 1.52, -2.25, 1.9, 1.42, 0.04, 0.0, -0.0],
                            ]
                        ]
                    ),
                    change_one_object_done=(
                        [
                            [('change_object_done', i)]
                            for i in [
                                'microwave',
                                'kettle',
                                'bottomknob',
                                'topknob',
                                'switch',
                                'slide',
                                'hinge',
                            ]
                        ]
                    ),
                ),
            ),
        ),
    ),
    buffer=dict(
        cls='DemonstrationsBuffer',
    ),
)
