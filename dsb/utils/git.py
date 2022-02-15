import subprocess


def get_commit():
    try:
        commit = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
    except:
        commit = None
    return commit


def is_repo_clean():
    try:
        clean = subprocess.check_output(['git', 'status', '--porcelain']).strip().decode() == ''
    except:
        clean = None
    return clean


def get_diff_patch():
    try:
        # subprocess.check_output(['git', 'add', '-A'])  # track all files
        patch = subprocess.check_output(['git', 'diff', '--patch', '--staged']).strip().decode()
        # subprocess.check_output(['git', 'reset'])  # unstage
    except:
        patch = ''
    return patch
