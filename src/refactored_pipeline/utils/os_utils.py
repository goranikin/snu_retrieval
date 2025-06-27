import os


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def remove_old_ckpt(dir_, k):
    ckpt_names = os.listdir(dir_)
    if len(ckpt_names) <= k:
        pass
    else:
        ckpt_names.remove("model_last.tar")
        steps = []
        for ckpt_name in ckpt_names:
            steps.append(int(ckpt_name.split(".")[0].split("_")[-1]))
        oldest = sorted(steps)[0]
        print("REMOVE", os.path.join(dir_, "model_ckpt_{}.tar".format(oldest)))
        os.remove(os.path.join(dir_, "model_ckpt_{}.tar".format(oldest)))
