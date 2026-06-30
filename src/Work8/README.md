# 计算机图形学 实验八：SMPL LBS线性混合蒙皮可视化


## 学生信息
202411998340 高小涵 计算机科学与技术专业


# 1. 项目架构
## (1)目录树
```
Work8/
├─ run_lbs_lab.py        # 主程序入口，完整实验全部逻辑
├─ models/
│  └─ smpl/
│     └─ SMPL_NEUTRAL.pkl # SMPL中性人体模型文件
└─ outputs/              # 所有可视化图片、误差日志输出目录
```
## (2)环境依赖
```bash
pip install torch torchvision smplx numpy matplotlib
```
- torch：张量计算、SMPL模型运算
- smplx：官方SMPL模型加载工具
- matplotlib/mplot3d：3D人体网格可视化
- numpy：数值处理、色彩映射

## (3)运行前置准备
1. 在`models/smpl/`下放SMPL_NEUTRAL.pkl模型文件；
2. 安装全部依赖库；
3. outputs文件夹程序自动创建，无需手动新建。

# 2. 代码逻辑 & 关键模块说明
## （1）工具辅助模块（全局通用函数）
| 函数 | 功能说明 |
|------|--------|
| `install_chumpy_pickle_shim()` | 兼容老式SMPL pickle文件，不安装chumpy也能加载模型 |
| `to_numpy()` | 张量/数组统一转numpy，适配matplotlib绘图 |
| `set_axes_equal()` | 3D坐标轴等比例缩放，避免人体网格拉伸变形 |
| `get_face_colors_from_vertex_scalar()` | 将顶点标量值映射面片色彩（权重/姿态偏移热力图核心） |
| `shade_face_colors()` | 简易漫反射光照，人体网格拥有立体光影效果 |
| `draw_mesh()` | 底层3D渲染封装，统一绘制网格、关节点、自定义色彩 |
| `save_single_figure()` | 保存单张阶段可视化图 |
| `save_comparison_grid()` | 2×2四阶段对比总图渲染 |
| `save_all_joint_weights_figure()` | 选做：全关节主导权重分布图生成 |

## （2）参数构造模块
1. `build_demo_shape()`：生成shape体型参数β，控制高矮胖瘦、肩宽腿长；
2. `build_demo_pose()`：生成姿态轴角参数θ，控制抬手、弯肘、屈膝、躯干扭转；
    - 内置关节名映射表，可精准修改单关节旋转角度。

## （3）核心：LBS全流程 `compute_manual_lbs()`
严格按照实验原理四阶段分步实现，输出5个核心中间变量：
1. **阶段A 模板网格**
    `v_template`：原始T-pose人体模板；`J_template`：模板姿态回归关节；
2. **阶段B 形状校正 Blend Shape**
    `v_shaped = v_template + blend_shapes(β, shapedirs)`
    `J_shaped = vertices2joints(J_regressor, v_shaped)` 从形变网格回归关节；
3. **阶段C 姿态校正 Pose Blend Shape**
    轴角→旋转矩阵 → pose_feature = R-I → pose_offsets姿态偏移量
    `v_posed = v_shaped + pose_offsets`；
4. **阶段D LBS线性混合蒙皮**
    层级刚体变换A → LBS权重W加权融合变换矩阵T → 齐次坐标计算最终人体顶点`verts`、变换后关节`J_transformed`。

## （4）误差校验模块 `compare_with_official_forward()`
- 输入完全相同β/姿态参数，调用smplx官方前向传播得到标准顶点；
- 逐顶点计算平均绝对误差、最大绝对误差，量化手写LBS与官方实现一致性。

## （5）主流程 `main()`
1. 命令行参数解析（模型路径、输出路径、可视化关节ID、β数量）；
2. SMPL模型加载，打印基础模型信息；
3. 生成shape/pose参数，执行手写LBS；
4. 执行官方前向对比，计算误差；
5. 分阶段渲染全部可视化图片；
6. 写入summary.txt保存模型信息与误差指标。

# 3. 实现功能
1. **任务1 SMPL模型加载与基础信息输出**
    加载中性SMPL模型，控制台输出顶点数、面片数、关节数、β维度，信息持久化存入summary.txt；
2. **任务2 模板网格与单关节蒙皮权重热力图**
    指定任意关节ID，将该关节对所有顶点的影响权重映射为网格颜色，生成stage_a_template_weights.png；
3. **任务3 形状校正与关节回归可视化**
    设置非零体型参数生成胖/瘦人体，从形变网格回归关节点，网格+关节同屏展示；
4. **任务4 姿态校正Pose Blend可视化**
    计算弯曲带来的姿态偏移量，用色彩表示偏移幅度，直观观察手肘、膝盖校正区域；
5. **任务5 完整LBS蒙皮结果渲染**
    输出经过骨骼加权变换后的最终姿态人体网格，附带变换后骨骼关节；
6. **任务6 四阶段对比总图**
    2×2子图一次性展示模板网格→体型形变→姿态校正→最终蒙皮全部中间状态；
7. **任务7 手写LBS与官方结果误差验证**
    计算MAE平均误差、Max最大误差，验证手写数学流程与官方实现完全对齐。


# 4. 可调参数与效果展示指南
运行代码支持通过命令行修改参数，不同参数会产生差异化可视化效果，下面给出可修改参数、运行命令、预期效果与截图。

## 4.1 核心可调命令行参数
| 参数 | 含义 | 修改效果 |
|------|------|---------|
| `--joint-id` | 指定可视化权重的关节编号 | 更换热力图显示的骨骼，如18=左肘、19=右肘、4=左膝、5=右膝 |
| `--num-betas` | 使用的体型参数β数量 | 数值越大，可调整体型细节越多，体型形变差异更明显 |
| `model_dir` | SMPL模型存放路径 | 模型文件路径修改时使用 |
| `out_dir` | 图片输出文件夹 | 自定义输出目录 |

## 4.2 测试方案1：更换可视化关节（权重热力图对比）
### 运行命令1（可视化左肘关节 joint=18）
```bash
python run_lbs_lab.py --joint-id 18
```
效果：stage_a_template_weights.png中左手臂、手肘区域色彩高亮，代表该区域顶点主要受左肘关节影响。
> 截图点：截取stage_a_template_weights.png，标注手肘区域高亮色彩。

### 运行命令2（可视化左膝关节 joint=4）
```bash
python run_lbs_lab.py --joint-id 4
```
效果：热力图集中在左腿膝盖附近，腿部权重颜色明显，躯干、手臂几乎无色彩。
> 截图点：两张热力图左右对比，说明不同骨骼控制人体不同区域皮肤。

## 4.3 测试方案2：修改体型参数维度（胖瘦形变对比）
### 运行命令1（默认10个β，中等体型变化）
```bash
python run_lbs_lab.py --num-betas 10
```
效果：stage_b_shaped_joints.png人体轻微偏壮，肩宽、躯干变厚；
### 运行命令2（使用20个β，更强体型形变）
修改代码`build_demo_shape`中β赋值幅度（如beta[0,0]=4.0放大体型变化），执行：
```bash
python run_lbs_lab.py --num-betas 20
```
效果：人体明显肥胖，躯干、四肢更粗壮，关节回归位置随体型同步外扩。
> 截图点：两张stage_b_shaped_joints.png对比，观察人体胖瘦、关节位置同步变化。

## 4.4 测试方案3：修改姿态参数（弯曲幅度对比）
修改`build_demo_pose()`内关节旋转轴角数值，例如：
1. 减小手肘旋转：`set_joint_pose("left_elbow", [0.0, -0.1, 0.0])` → 手臂微弯
2. 增大手肘旋转：`set_joint_pose("left_elbow", [0.0, -0.8, 0.0])` → 手臂大幅弯折
效果对比：
- stage_c_pose_offsets.png：手肘弯曲越大，肘部姿态校正偏移量色彩越深；
- stage_d_lbs_result.png：最终人体手臂弯曲幅度差异巨大。
> 截图点：对比两张姿态校正图、两张最终LBS效果图，说明pose offsets用于修正骨骼弯曲皮肤形变。

## 4.5 误差指标效果说明
修改任意参数后，summary.txt内`manual_vs_official_mean_abs_error`始终维持极小值（1e-6量级），证明手写LBS数学流程和官方实现完全等价；
- 若参数正确加载，平均误差<1e-5；
- 截图点：打开summary.txt截取误差数值，作为实验验证依据。

# 5. 输出文件说明
运行完成后outputs目录生成全部实验要求文件：
1. `stage_a_template_weights.png`：任务2单关节权重热力图
2. `stage_b_shaped_joints.png`：任务3体型校正+回归关节
3. `stage_c_pose_offsets.png`：任务4姿态校正偏移热力图
4. `stage_d_lbs_result.png`：任务5完整LBS最终人体
5. `comparison_grid.png`：任务6四阶段2×2总对比图
6. `all_joint_weights.png`：选做全关节主导权重分布图
7. `summary.txt`：模型基础信息、手写LBS与官方误差指标

# 6. 实验思考题简答（可附在报告末尾）
1. 为什么一个顶点不只受一个关节影响？
    人体关节连接处（肩、膝、肘）皮肤会同时跟随两根骨骼运动，单关节权重会出现硬折痕，多关节加权混合实现平滑过渡。
2. 如果顶点权重几乎全给单一关节：关节弯折处皮肤会生硬撕裂、出现明显棱角，无平滑过渡。
3. 权重分布平均：关节运动时周边皮肤同步拉扯，形变柔和但骨骼控制力度弱，动作僵硬。
4. 关节需要从v_shaped回归：胖瘦体型会改变骨骼实际空间位置，固定关节无法适配不同体型人体。
5. 去除pose_offsets：手肘、膝盖弯曲时皮肤会穿模、凹陷，无法模拟人体肌肉挤压形变。
