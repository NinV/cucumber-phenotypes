import pathlib
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preparing image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root_dir', default='agri_phenotyping')

    parser.add_argument(
        '-t', '--template', type=str, default='board_template_images/A1-v1/A1-0004-v1.jpg',
        help='Path to the template image')

    parser.add_argument(
        '-i', '--img_dir', type=str, default='cucumber',
        help='Path to the directory that contains the images')

    parser.add_argument('-o', '--output', type=str, default='agri_phenotyping/pairs.txt')
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = pathlib.Path(args.root_dir)
    img_dir = root_dir / args.img_dir
    template = root_dir / args.template

    img_files = list(img_dir.glob("*.jpg"))
    print('Found {} image files at "{}"'.format(len(img_files), img_dir))
    with open(args.output, 'w') as f:
        for imf in img_files:
            f.write('{},{}\n'.format(str(imf.relative_to(root_dir)), args.template, ))


if __name__ == '__main__':
    main()
