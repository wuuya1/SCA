import json
import os
import glob
import imageio
import numpy as np
import pandas as pd
from plt3d import plot_episode


case_name = 'sca'       # Update the file name for visualization trajectories in different policies.

abs_path = os.path.abspath('.')
plot_save_dir = abs_path + '/' + case_name + '/'
log_save_dir = abs_path + '/' + case_name + '/'


def get_agent_traj(info_file, traj_file):
    with open(info_file, "r") as f:
        json_info = json.load(f)

    traj_list = []
    step_num_list = []
    agents_info = json_info['all_agent_info']
    obstacles_info = json_info['all_obstacle']
    for agent_info in agents_info:
        agent_id = agent_info['id']

        df = pd.read_excel(traj_file, index_col=0, sheet_name='agent' + str(agent_id))
        step_num_list.append(df.shape[0])
        traj_list.append(df.to_dict('list'))

    return agents_info, traj_list, step_num_list, obstacles_info


def png_to_gif(general_fig_name, last_fig_name, animation_filename, video_filename):
    all_filenames = plot_save_dir + general_fig_name
    last_filename = plot_save_dir + last_fig_name
    print(all_filenames)

    # Dump all those images into a gif (sorted by timestep).
    filenames = glob.glob(all_filenames)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
    for i in range(10):
        images.append(imageio.imread(last_filename))

    # Save the gif in a new animations sub-folder.
    animation_save_dir = plot_save_dir + "animations/"
    os.makedirs(animation_save_dir, exist_ok=True)
    video_filename = animation_save_dir + video_filename
    animation_filename = animation_save_dir + animation_filename
    imageio.mimsave(animation_filename, images, )

    # Convert .gif to .mp4.
    try:
        import moviepy.editor as mp
    except imageio.core.fetching.NeedDownloadError:
        imageio.plugins.ffmpeg.download()
        import moviepy.editor as mp
    clip = mp.VideoFileClip(animation_filename)
    clip.write_videofile(video_filename)


info_file = log_save_dir + 'log/env_cfg.json'
traj_file = log_save_dir + 'log/trajs.xlsx'

agents_info, traj_list, step_num_list, obstacles_info = get_agent_traj(info_file, traj_file)

agent_num = len(step_num_list)
base_fig_name_style = "{test_case}_{policy}_{num_agents}agents"
base_fig_name = base_fig_name_style.format(policy=case_name, num_agents=agent_num, test_case=str(0).zfill(3))
general_fig_name = base_fig_name + '_*.png'
last_fig_name = base_fig_name + '.png'
animation_filename = base_fig_name + '.gif'
video_filename = base_fig_name + '.mp4'

plot_episode(obstacles_info, agents_info, traj_list, step_num_list, plot_save_dir, base_fig_name, last_fig_name)

png_to_gif(general_fig_name, last_fig_name, animation_filename, video_filename)


