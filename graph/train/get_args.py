import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train Graph model')
    parser.add_argument('--config', type=str, default='', help='config file')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    args = parser.parse_args()
    print(args)
    return args