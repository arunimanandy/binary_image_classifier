import argparse, random, shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import yaml

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def collect_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in VALID_EXTS])


def resize_rgb_image(src, dst, size=224):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert('RGB').resize((size, size))
        img.save(dst, format='JPEG', quality=95)
    return dst


def split_files(files, train_ratio, val_ratio, test_ratio, seed):
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError('Split ratios must sum to 1.0')
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]


def preprocess_class(files, label, processed_dir, image_size, split_name):
    output = []
    for idx, src in enumerate(files):
        dst = Path(processed_dir) / split_name / label / f'{label}_{idx:06d}.jpg'
        try:
            output.append(resize_rgb_image(src, dst, image_size))
        except (UnidentifiedImageError, OSError):
            continue
    return output


def run(config_path):
    cfg = load_config(config_path)
    data_cfg = cfg['data']
    processed_dir = Path(data_cfg['processed_dir'])
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    image_size = int(cfg['image_size'])
    seed = int(cfg['seed'])
    summary = {}
    for label, folder_key in [('cat','cat_dir'), ('dog','dog_dir')]:
        files = collect_images(data_cfg[folder_key])
        if not files:
            raise FileNotFoundError(f'No images found for {label} in {data_cfg[folder_key]}')
        train, val, test = split_files(files, data_cfg['train_ratio'], data_cfg['val_ratio'], data_cfg['test_ratio'], seed)
        summary[(label,'train')] = len(preprocess_class(train, label, processed_dir, image_size, 'train'))
        summary[(label,'val')] = len(preprocess_class(val, label, processed_dir, image_size, 'val'))
        summary[(label,'test')] = len(preprocess_class(test, label, processed_dir, image_size, 'test'))
    print('Preprocessing complete:', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs.yaml')
    args = parser.parse_args()
    run(args.config)
