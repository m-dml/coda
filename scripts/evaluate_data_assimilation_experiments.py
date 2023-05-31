import os

observations_length = 250

search_directories_list = [
    "/work/bg1315/Zinchenko/output/data_assimilation",
]
n_trials = 1000


def get_query_string(
    directories_list_string: str,
    observations_length: int,
    n_trials: int,
) -> str:
    query_string = " ".join(
        (
            "python evaluate_data_assimilation.py",
            f"exp_base_dir={directories_list_string}",
            f"observations_length={observations_length}",
            f"n_trials={n_trials}",
            "hydra/launcher=local -m",
        )
    )
    return query_string


def find_experiments_directories(directory_list: list[str]) -> list[str]:
    direcories_list = []
    for directory in directory_list:
        for root, dirs, _ in os.walk(directory):
            for subdir in dirs:
                if ".hydra" in subdir:
                    path = os.path.join(root, subdir)[:-7]
                    direcories_list.append(path)
    return direcories_list


def main():
    os.chdir("..")

    directories_list = find_experiments_directories(search_directories_list)
    directories_list_sting = ",".join(directories_list)
    shell_command = get_query_string(directories_list_sting, observations_length, n_trials)
    os.system(shell_command)


if __name__ == "__main__":
    main()
