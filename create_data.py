from pathlib import Path
import shutil

p = Path('frames') / 'F1_1_1_1'
train_dir = Path('input') / 'train'
test_dir = Path('input') / 'test'

paths = list(sorted(p.iterdir(), key=lambda path: int(path.stem.rsplit(".", 1)[0])))
test_size = len(paths) // 5
for name in paths[:-test_size]:
    shutil.copy2(name, train_dir / name.name)
for name in paths[-test_size:]:
    shutil.copy2(name, test_dir / name.name)