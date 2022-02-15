from dsb.dependencies import *
from dsb.utils import save_video, torchify, untorchify
from mpl_toolkits.mplot3d import Axes3D

from ..buffer_wrapper import BufferWrapper


class EpisodeVideoRecorderBufferWrapper(BufferWrapper):
    def __init__(
        self,
        buffer,
        ckpt_dir='',
        episode_log_interval=100,
        with_goal=True,
        with_goal_context=True,
        with_goal_context_diff=True,
        with_recon=True,
        with_saliency=False,
        with_action_vector=False,
        with_action_log=False,
        action_indices=[[0, 1, 2], [7, 8]],
        compute_recon_func=None,
        compute_saliency_func=None,
        **kwargs,
    ):
        super().__init__(buffer, **kwargs)
        self.num_episodes = 0
        self.ckpt_dir = ckpt_dir
        self.episode_log_interval = episode_log_interval
        self.with_goal = with_goal
        self.with_goal_context = with_goal_context
        self.with_goal_context_diff = with_goal_context_diff
        self.with_recon = with_recon
        self.with_saliency = with_saliency
        self.with_action_vector = with_action_vector
        self.with_action_log = with_action_log

        self.action_indices = action_indices
        self.compute_recon_func = compute_recon_func
        self.compute_saliency_func = compute_saliency_func

    def plot_action_vector(self, actions, img_size):
        plots = []
        for action in actions:
            action_plot = []

            for idx in self.action_indices:
                fig = plt.figure()
                DPI = float(fig.get_dpi())
                # render at 2x the desired resolution
                render_factor = 2
                fig.set_size_inches(
                    render_factor * img_size[1] / DPI, render_factor * img_size[0] / DPI
                )  # figure expects (width, height)

                ax = None
                if len(idx) == 3:
                    ax = fig.add_subplot(111, projection='3d')

                    ax.quiver(*[0, 0, 0], *action[idx])
                    ax.scatter(*[0, 0, 0])
                    ax.set_title(
                        f'a= {np.round(action[idx], 2)}\n|a|= {np.linalg.norm(action[idx]):.2f}'
                    )

                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_zlim(-1, 1)
                elif len(idx) == 2:
                    ax = fig.add_subplot(111)

                    ax.arrow(*[0, 0], *action[idx])
                    ax.set_title(
                        f'a= {np.round(action[idx], 2)},|a|= {np.linalg.norm(action[idx]):.2f}'
                    )

                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                else:
                    raise ValueError

                fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,)
                )
                image_from_plot = cv2.resize(image_from_plot, img_size)
                action_plot.append(image_from_plot)

            action_plot = np.concatenate(action_plot, axis=1)
            plots.append(action_plot)

            # from PIL import Image
            # Image.fromarray(action_plot).save('a.png')
            # exit()
        return np.stack(plots)

    def record_video(self, ep_ptrs):
        ep_frames = self.buffer.get_state(ep_ptrs, 'achieved_goal').copy()
        ep_goals = self.buffer.get_state(ep_ptrs, 'desired_goal').copy()

        video = ep_frames.transpose((0, 2, 3, 1))  # convert to channel last

        # RGB or grayscale

        if self.with_goal:
            # concatenate on right side, horizontal
            video = np.concatenate([video, ep_goals.transpose((0, 2, 3, 1))], axis=2)

        if self.buffer.with_context:
            ep_goals_context = None
            if self.with_goal_context:
                if ep_goals_context is None:
                    ep_goals_context = {}
                    for k in self.buffer.context_type.keys():
                        ep_goals_context[k] = self.buffer.get_state(ep_ptrs, f'c_{k}').copy()

                for k, v in ep_goals_context.items():
                    video = np.concatenate([video, v.transpose((0, 2, 3, 1))], axis=2)

            if self.with_goal_context_diff:
                if ep_goals_context is None:
                    ep_goals_context = {}
                    for k in self.buffer.context_type.keys():
                        ep_goals_context[k] = self.buffer.get_state(ep_ptrs, f'c_{k}').copy()

                for k, v in ep_goals_context.items():
                    if k == 'desired_goal':
                        d = ep_goals - v
                    elif k == 'achieved_goal':
                        d = ep_frames - v
                    video = np.concatenate([video, d.transpose((0, 2, 3, 1))], axis=2)

        if self.with_recon:
            # TODO: check if recon requires context
            recon = self.compute_recon_func(torchify(ep_frames.copy()))
            recon = untorchify(recon)
            recon = recon.astype(np.uint8)
            video = np.concatenate([video, recon.transpose((0, 2, 3, 1))], axis=2)

            # TODO: add goal recon

        # check if grayscale and convert to RGB
        if video.shape[3] == 1:
            video = video.repeat(3, axis=-1)
            # video_frames = []
            # for frame in video:
            #     video_frames.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
            # video = np.stack(video_frames)

        # RGB

        if self.with_saliency:
            s = dict(achieved_goal=ep_frames, desired_goal=ep_goals)
            gradcam, guided_backprop, guided_gradcam, heatmap = self.compute_saliency_func(s)

            for x in [gradcam, guided_backprop, guided_gradcam, heatmap]:
                for k, v in x.items():
                    if v.shape[1] == 1:  # if grayscale, then convert to rgb
                        v = v.repeat(3, axis=1)
                    if v.dtype != np.uint8:
                        v = (v * 255.0).astype(np.uint8)
                    x[k] = v
                x = np.concatenate([x[k] for k in s.keys()], axis=-1)
                video = np.concatenate([video, x.transpose((0, 2, 3, 1))], axis=2)

        ep_actions = self.buffer.action[ep_ptrs].copy()

        if self.with_action_vector:
            action_vector_plots = self.plot_action_vector(ep_actions, ep_frames[0].shape[1:])
            video = np.concatenate([video, action_vector_plots], axis=2)

        if self.with_action_log:
            pos = (0, 10)
            bg_color = (255, 255, 255)

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.25
            color = (0, 0, 0)
            thickness = cv2.FILLED
            margin = 2

            for i in range(len(ep_ptrs)):
                frame_action = ep_actions[i]
                frame = video[i]

                text = str(np.round(frame_action, 2))
                txt_size = cv2.getTextSize(text, font_face, scale, thickness)
                end_x = pos[0] + txt_size[0][0] + margin
                end_y = pos[1] - txt_size[0][1] - margin

                cv2.rectangle(frame, pos, (end_x, end_y), bg_color, thickness)
                cv2.putText(frame, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

        save_video(
            os.path.join(self.ckpt_dir, 'episode_videos', f'video_ep_{self.num_episodes}.avi'),
            video,
            fps=5,
        )

    def add(self, *transition, last_step=False, env_idx=None):
        ptr = self.buffer.add(*transition, last_step=last_step, env_idx=env_idx)

        if last_step:
            if self.num_episodes % self.episode_log_interval == 0:
                # get episode ptrs and save a video
                ep_ptrs = self.episode_ptrs[self.start_ptr[ptr]][:]

                self.record_video(ep_ptrs)
                # exit()

            self.num_episodes += 1
        return ptr
