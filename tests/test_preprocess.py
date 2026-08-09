from pathlib import Path
from PIL import Image
from src.data.preprocess import resize_rgb_image, split_files


def test_resize_rgb_image(tmp_path):
    src = tmp_path / 'input.png'
    Image.new('RGBA', (20, 30), (255, 0, 0, 128)).save(src)
    dst = tmp_path / 'out' / 'image.jpg'
    resize_rgb_image(src, dst, size=224)
    with Image.open(dst) as img:
        assert img.mode == 'RGB'
        assert img.size == (224, 224)


def test_split_files_ratios():
    files = [Path(f'{i}.jpg') for i in range(100)]
    train, val, test = split_files(files, 0.8, 0.1, 0.1, 42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
