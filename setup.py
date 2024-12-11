import os
import subprocess


def install_requirements():
    if not os.path.exists('requirements.txt'):
        raise FileNotFoundError("requirements.txt not found")
    print("\033[92mInstalling requirements...\033[0m")
    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])


def check_and_download_data():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, 'train')) or \
       not os.path.exists(os.path.join(data_dir, 'dev')) or \
       not os.path.exists(os.path.join(data_dir, 'test')):
        print("\033[92mNot Found PhoMT data, downloading ...\033[0m")
        subprocess.check_call(['gdown', '--folder', '1cPdLNnTlsj3N1FE9x6_K608bCAaYaVGM', '-O', 'data'])
    else:
        print("\033[92mFound PhoMT data\033[0m")


def install_more():
    print("\033[92mInstall more\033[0m")

    import nltk
    nltk.download('punkt_tab')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Setup script for machine translation project')
    parser.add_argument('--skip-install', action='store_true', help='Skip installing requirements')
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()
    
    check_and_download_data()
    install_more()
    print("\033[92mDone\033[0m")