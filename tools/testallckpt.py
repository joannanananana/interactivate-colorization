import os
import subprocess

ckpt_folder = "./work_dirs/GandD_finetune/ckpt/GandD_finetune/"
demo_template = "python demo/video_demo.py shangganling_cut_1min.mp4 configs/deoldify/deoldify_nogan_GandD_finetune.py ckpt_path --out ./work_dirs/ckpt_name.mp4"


def run_demo(ckpt_path, ckpt_name):
    demo_command = demo_template.replace("ckpt_path", ckpt_path)
    demo_command = demo_command.replace("ckpt_name", ckpt_name)
    print(demo_command)
    subprocess.run([demo_command], shell=True)


for cur, dirs, files in os.walk(ckpt_folder):
    for f in files:
        ckpt_path = ckpt_folder + f
        run_demo(ckpt_path, f)
