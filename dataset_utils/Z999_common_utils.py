import os
import shutil


def remove_module_dir(module_output_dir):
    if os.path.exists(module_output_dir):
        shutil.rmtree(module_output_dir)