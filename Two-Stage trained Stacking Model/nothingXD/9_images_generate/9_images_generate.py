import matplotlib.pyplot as plt
from PIL import Image

# 图片路径列表
image_paths = ['images/Electricity-XGBoost.png', 'images/Electricity-SVR.png', 'images/Electricity-Stacking.png',
               'images/Temp-XGBoost.png', 'images/Temp-SVR.png', 'images/Temp-Stacking.png',
               'images/Exchange-XGBoost.png', 'images/Exchange-SVR.png', 'images/Exchange-Stacking.png']

# 创建一个新的大图（这里假设每个图像的宽为w，高为h）
w, h = Image.open(image_paths[0]).size
grid = Image.new('RGB', (w * 3, h * 3))

# 逐个粘贴图片到大图上
for index, path in enumerate(image_paths):
    img = Image.open(path)
    # 计算此图在大图上的位置
    x = index % 3 * w
    y = index // 3 * h
    grid.paste(img, (x, y))

# 保存大图
grid.save('grid_image.png')

# 显示大图
#grid.show()
