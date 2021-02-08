import os

def lookup_env_filename(data_loc, env_idx):
    save_dir = 'models/' + data_loc + '/gt_policies/'
    filename = save_dir + 'wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'

    return filename

def list_env_filenames(data_loc):
    save_dir = 'models/' + data_loc + '/gt_policies/'
    filenames = sorted(os.listdir(save_dir))

    return filenames

def attempt_pool_restart(pool):
    try:
        pool.restart()
    except:
        pass

