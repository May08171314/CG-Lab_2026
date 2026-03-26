# 计算机图形学基础实验 - 3D 变换与渲染
基于 Taichi 实现的计算机图形学基础实验，包括 3D 坐标变换、透视投影、MVP 矩阵、3D 立方体渲染。
## 1. 项目架构
项目采用 Python标准源码布局（src源码结构），将源码统一放在src/目录下，便于包管理、导入控制和项目维护。

项目目录结构如下：
```markdown
CG_Lab_2026/
├── pyproject.toml
├── src/
│   └── Work2/
│       ├── triangle.py       --> 3D 三角形渲染
│       ├── cube_line.py      --> 3D 线框立方体
│       ├── cube_face.py      --> 3D 彩色面立方体
│       └── README.md      
```
环境管理上，基于 uv 管理虚拟环境与依赖。（该部分内容不在仓库存储范围内）

## 2. 运行方式
### （1） 代码运行
#### a. 运行 `triangle.py` 文件
```python
uv run -m src.Work2.triangle
```
#### b. 运行 `cube_line.py` 文件
```python
uv run -m src.Work2.cube_line
```
#### c. 运行 `cube_face.py` 文件
```python
uv run -m src.Work2.cube_face
```
### （2）操作说明
#### a. 3D 三角形
- A/D：旋转三角形
#### b. 3D 线框立方体
- A/D：左右旋转
- W/S：上下旋转
#### c. 3D 彩色面立方体
- A/D：左右旋转
- W/S：上下旋转
## 3. 实现功能
### （1）`triangle.py`
- 完整 MVP 矩阵变换
- 透视投影
- 绕Z轴旋转
- 3D 空间坐标 → 屏幕坐标映射
### （2）3D 线框立方体
- 双轴欧拉角旋转（X + Y）
- W/A/S/D 流畅控制
- 透视投影
### （3）3D 彩色面立方体
- 双轴欧拉角旋转（X + Y）
- W/A/S/D 流畅控制
- 6 个彩色面渲染
## 4. 效果展示

## 5. 可调参数
- 所有程序均可自由调整：
- 旋转速度
- 模型大小
- 线条粗细 / 颜色
- 相机距离
- 透视视角（FOV）
- 窗口大小
