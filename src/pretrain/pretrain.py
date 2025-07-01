import os

def get_argv(result_dir) -> list[str]|None:
    stat_path = f"{result_dir}/stats.txt"
    if not os.path.exists(stat_path):
        return None
    with open(stat_path) as f:
        argv = f.readline()
    return argv.split(' ')

def get_scheme(result_dir):
    argv = get_argv(result_dir)
    if argv is None: return None
    script = argv[0]
    if script == 'VICRegL':
        return 'VICRegL'
    elif 'vicreg' in script:
        return 'vicreg'
    else:
        return 'barlowtwins'
