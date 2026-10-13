# Transcraft

手搓 Transformer 学习环境：Python 练习加浏览器可视化画布，拖拽节点即可生成并运行 PyTorch 代码。

## 可视化画布（拖拽生成 Python 代码）

模型页提供 `__init__` / `forward` 两个生命周期页签：`__init__` 展示构造参数及可编辑注册成员列表，`forward` 展示数据流画布。旧图的层/子模型直接映射到成员列表，不复制、不重命名、不删除节点或连线；当前不是两个独立 Drawflow 画布。

`canvas/` 目录是一个学习辅助工具：在网页上拖拽节点、连线，后台把画布翻译成各项目独立的 `canvas/projects/<id>/generated_model.py`。

```bash
# 启动（在项目目录下，默认端口 8000）
.venv/Scripts/python.exe canvas/server.py
# 浏览器打开 http://127.0.0.1:8000
# 端口被占用时换端口启动：CANVAS_PORT=8002 .venv/Scripts/python.exe canvas/server.py
```

- **基础值**：左侧「数据 / 输入」中的「基础值 Literal」，在节点内常显的「Python 字面量」参数栏填写数字、带引号字符串、`True` / `False`、`None`、列表或元组。后端 `/api/literal` 使用 `ast.literal_eval` 校验并规范化表达式，拒绝函数调用、推导式及字典/集合；最大 10000 字符。生成普通 Python 值，可连接运算、打印和函数参数，支持主画布、模型 forward、普通函数及存档恢复。
- **变量引用**：左侧「数据 / 输入」中的「变量引用 Variable Reference」没有输入端口，只有一个输出。在节点内常显的「变量名」参数栏直接填写当前作用域的实际变量名（支持合法中文名），例如 `dim`、`x`、`weights`、调用多返回值的 `result_out2`。编译器从实际输出符号表解析，不执行表达式；拒绝 `x[0]`、`x + 1`、任意属性或函数调用。引用生成独立的 `ref = weights` 别名，不覆盖上游变量；即使上游节点后创建也自动按依赖排序，缺失、歧义或循环会保存草稿但阻止运行。主画布、普通函数和模型 forward 各自隔离，不捕获其他模型或全局变量。模型构造参数 `dim` 只有在 forward 引用 `dim` 或 `self.dim` 时才生成 `self.dim = dim`；Linear/Embedding 尺寸直接使用构造参数的原行为不变。`self.<成员>` 仅接受此模型已注册的成员；模型实例别名只能接调用实例端口。引用运行值保持原对象和梯度，来源显示对应生产者/输出端口，构造参数显示初始化代码行。复制引用节点只产生新别名，保留明确填写的目标变量。Literal 中原有的 `dim` 不会自动替换，需自行换成引用节点。
- **节点与端口**：紧凑彩色标题就是节点变量名（如 `WQ`），点击标题原位编辑，Enter 或失焦提交、Esc 取消、空值不生效；模型输入节点的标题即 forward 参数名，编辑写入 `argname`。标题下方显示实际调用目标（如 `nn.Linear · 线性变换`、`ShankeModel.forward`）及参数摘要，仅此一份。标题和摘要自动换行，节点宽度固定为 260px；端口位置跟随实际行中心，长端口名可悬停查看。输入左侧、输出右侧逐行排列，空心/实心区分未连接/已连接。端口显示真实参数名，基础值显示校验后的 Python 类型；动态参数未知类型显示 `Any`，不假称为 Tensor。
- **节点复制**：右键节点选择「复制节点」，再右键画布或节点选择「粘贴节点」，在点击位置创建独立节点。使用应用内部单节点剪贴板，不读写系统剪贴板；保留参数、接口端口数量及详情展开状态，不复制连线、标识符或运行值，新名称加 `_copy` 后缀并自动查重。模型目标必须仍存在，输入/返回节点不能粘贴到主画布；切换项目清空剪贴板。菜单支持方向键、Enter、Esc 及点击外部关闭；输入框和代码文本仍使用原生复制。
- **节点详情与分类**：基础节点分为「数据 / 输入」「模型层」「数学运算」「张量变换」「遮罩」「输出」，各组默认展开，可用鼠标或键盘折叠。模型/函数定义独立列出。所有节点参数始终直接显示在节点内编辑（dtype 等仍为下拉选择控件），不再点开抽屉、没有参数/运行结果页签；参数区与运行结果各自独立滚动，节点宽度始终保持 260px。标题右侧箭头只负责展开/收起运行结果，多个节点的结果可同时展开；点击空白取消选中代码高亮但保留表单；切换画布保留独立展开状态，模型 `__init__` 页收起该模型的运行结果。输入、按钮和抽屉文本选择不会拖动节点，标题在非编辑状态仍可拖拽节点。右侧仅保留代码预览。模型构造参数仅在 `__init__` 页设置，成员标题可跳转到其 forward 节点。
- **基础节点**：Tensor 输入、nn.Embedding、nn.Linear、矩阵乘 @、运算 op（2–16 个输入，行间运算符，见下条）、转置、Softmax、打印、打印形状、索引取值、`torch.ones_like`、`torch.triu`、`Tensor.to`（默认 bool）、`Tensor.masked_fill`。
- **运算 op 多输入与运算顺序**：运算 op 默认两个输入 a、b，用节点内「＋ 增加输入 / － 移除最后输入」可扩到 2–16 个（c、d…）。每个输入行的编辑框可直接手填字面量（走 Literal 安全校验）；该输入连了线就用上游值，行内编辑框禁用并注明「来自上游连线」，断开后自动恢复手填值。相邻输入行之间的下拉框是运算符，新输入继承上一段运算符。计算严格从上至下左折叠并加显式括号：输入 a、b、c 配 `+`、`*` 生成 `result = ((a + b) * c)`，与 Python 自身优先级无关；两个输入保持 `result = a op b`。输入既未连线也未填值、运算符为空或非法、手填字面量非法都会明确报错（不会静默按 0 计算），报错时草稿可保存但不能运行。移除已连线的末端输入会先弹页面内确认，只删该输入及其连线，不影响前面连线。
- **遮罩参数**：`triu.diagonal` 支持负数、0、正数，作用于最后两个维度；`masked_fill` 的 bool mask 为 True 时填充，False 时保留。diagonal 和填充值各有可选输入端口，连接 Literal 标量时连线优先，否则使用面板静态值。填充值支持数字及 `-inf` / `inf` / `nan`；Literal 本身仍拒绝非有限字面量。`ones_like` 的 dtype/device 可选择 inherit、常用类型及 cpu/cuda，选择 CUDA 需要实际可用 GPU；这些参数不支持动态连线。参数旁的 `?` 提供局部解释，非法 mask 类型、广播、整数填 -inf 等由真实 PyTorch 报错，不静默转换。
- **模型类**：顶部「＋ 模型」创建模型。在模型页设置 `__init__` 参数（例如 `vocab_size, embedding_dim`），Embedding/Linear 的尺寸可引用这些参数。模型输入/输出声明 `forward`。层及子模型在 `__init__` 注册，每个计算节点在前向中只执行一次并保存为变量。
- **显式实例**：左侧分别提供「创建模型实例」和「调用已有实例」。创建节点填写位置构造实参（整数或外层构造参数，逗号分隔），将 instance 输出接到调用节点的 instance 输入；同一创建节点可连接多个调用，复用权重。其余输入连接 Tensor。
- **普通函数**：「+ 函数」创建 `def` 定义，不继承 `nn.Module`。函数输入/输出声明接口，左侧提供普通函数调用。
- **兼容与校验**：v2 旧图不删除节点，旧调用保留“创建并调用”语义；模型内的旧调用现在始终注册为子模块（包括多输出）。首次 v3 保存会备份旧文件至 `canvas/backups/before-first-v3-save/`。接口增加端口自动追加；缩减端口不丢弃旧连线，显示「废弃端口」警告并阻止生成可运行代码（草稿仍可保存）。确认原连线并重建调用节点后恢复运行。此次界面更新前原文件备份在 `canvas/backups/before-blueprint-1791622815425/`。
- **节点运行值**：运行后点标题箭头展开「运行结果」，在节点内部显示 shape、dtype、device、实际数值矩阵及行列索引；高维张量可在捕获的起始区域内选择 batch/head 等前置维度。输入和返回节点、嵌套模型、普通函数、实例调用及多输出分别捕获，调用样本与输出下拉框可切换。非张量显示类型及有限内容，模型实例仅显示短类型，不展开权重。来源面板列出输入连线和对应代码行，可打开上游节点的运行结果抽屉；模型/函数输入还显示该调用样本在外层画布的实际输入来源，可跨页打开。现有 Prism 预览和选中代码高亮保留。
- **追踪实现与边界**：`generated_model.py` 保持普通 PyTorch 源码；服务在独立子进程中对其 AST 做仅执行时的插桩，原张量和返回值不替换，只在预览时 detach。每次运行在项目 `runs/<run-id>.jsonl` 写结构化记录，不从 stdout 解析。预览先切片再复制 CPU，每张量最多 2048 值，最后两维各最多 16，前置维各最多 2；整个运行最多 32768 值及 512 记录，每记录最多 16 输出。稀疏、量化、meta 或超过 16 维的张量不展示数据。NaN/无穷与超出 JS 精确整数范围的数值序列化为字符串，文本通过 textContent 渲染。输出文本最多 64 KiB。运行失败或超时保留此前记录并显示错误；图编辑、项目/保存 revision 不匹配时标记「已过期」，禁止跳转旧来源。非捕获区域无法按需读取，运行文件目前无自动清理；追踪会增加执行开销，不是训练或安全沙箱。
- **限制**：构造实参暂不支持任意表达式和关键字参数。普通函数中的层按 Python 函数语义每次执行时创建。打印节点没有输出端口，因此只在底部展示打印内容。
- **保存保护**：未完成的图也可自动或手动保存为草稿；Python 语法或字面量校验失败不会替换最后通过校验的代码，也不会运行旧代码。仅本机可信代码使用，张量/索引表达式不是安全沙箱。
- **实时预览**：右侧面板同步显示生成的 Python 代码；点「▶ 运行」用项目 .venv 执行并在底部显示输出
- **项目**：顶部下拉框打开项目，「新建」「重命名」「保存」「另存为」管理独立画布。名称去除首尾空格，不区分大小写查重；名称不是路径。目录使用不可变 32 位随机 ID。切换前等待当前草稿落盘，保存冲突或网络失败阻止切换；另存为可把本地修改保留到新项目。
- **磁盘格式**：`canvas/projects/<id>/project.json`（schema 1）是原子提交的权威快照；`graph.json` 为最新草稿，`generated_model.py` 为最后通过生成与语法校验的代码。空白新项目在首次有效生成后才有 Python 文件。`versions/<version>.json` 保留各次完整快照，尚无历史恢复 UI，也没有自动清理策略。磁盘中断时镜像可能领先权威快照，服务读取和执行仅依据 `project.json`。
- **迁移与恢复**：首次打开项目列表会把旧 `canvas/graph.json` 和 `generated_model.py` 非破坏性复制至 Default 项目，并在该项目 `legacy/` 保留逐字节原件；旧文件不再写入。损坏旧图会报错，不加载空图或示例。此次修改前备份位于 `canvas/backups/before-projects-20261010/`。刷新默认打开 Default；浏览器未同步草稿存储于 `localStorage`，恢复时另建项目，保留 `canvas-recovery:*` 副本。更新前已打开的旧页面必须先由浏览器导出当前图，不能直接刷新而丢失尚未写盘的数据。
- **浏览器草稿保护**：「导出备份」下载当前图、项目及 canvas 草稿/恢复 localStorage。项目命名和草稿恢复使用页面内对话框，不依赖浏览器 prompt/confirm。拒绝恢复后自动保存暂停时，可直接点「恢复草稿」另存为独立项目，无须刷新，旧项目和原浏览器草稿不覆盖。本次源码、磁盘数据和更新前浏览器状态备份在 `canvas/backups/before-runtime-20261010/`；`browser-state.json` 含实际更新前图、state 和 storage，验收用 `causal-viewer-demo.json` 独立于用户项目。
- **版本与执行**：保存、重命名和运行要求项目 ID 与整数版本一致；并发标签页冲突返回 409。运行还要求对应本次有效保存的随机 revision；草稿或重命名会撤销旧执行令牌。执行捕获该项目代码及工作目录，不会借用其他项目文件。服务只绑定 `127.0.0.1`，不是 Python 沙箱，不提供训练功能。
- 顶栏「示例」按钮：第 1 课 ShankeModel（类版完整复现）、自注意力 SelfAttention 类

## 环境

回归测试：变量引用测试已纳入 `node canvas/test_generator.cjs` 与 `node canvas/test_runtime.cjs`，覆盖构造参数提升/无引用不提升、声明 self 成员、中文参数、普通函数与 forward 参数、主画布多返回值、隐式拓扑依赖/循环/缺失/表达式拒绝、独立别名与复制目标、后端 trace plan 校验、来源及梯度一致性；全部使用内存图，不改动活动项目。`node canvas/test_generator.cjs`（需要项目 `.venv/Scripts/python.exe` 与 PyTorch）。包含 ShankeModel 构造参数、重复实例调用、多输出嵌套模型、普通函数、序列化恢复、非法图、SelfAttention 数值/形状断言，以及基础值类型/执行/恢复、未校验值拒绝和旧端口保留。还包含离线 DOM 合约测试：分类组结构、右侧仅代码、常显参数随新增/导入/粘贴挂载且不重复绑定、标题编辑提交/取消/空值/argname、关闭运行结果不清参数、同 ID 跨画布独立表单与运行值、多个结果同时展开、变量名标题及调用摘要、缩放后的端口行中心、内部剪贴板独立复制/端口保留/新 ID/作用域和项目隔离/菜单行为、共享表单的参数帮助及控件、节点重命名/参数编辑后代码真实执行、高维切片、过期来源禁用及上游抽屉跳转。后端隔离测试：`.venv/Scripts/python.exe canvas/test_server.py`，验证项目隔离、非破坏迁移、项目绑定运行、版本冲突、草稿保留代码、重命名查重、非法输入不覆盖及字面量 API。`node canvas/test_projects.cjs` 验证保存排队、版本捕获、无效草稿、异步旧响应和冲突时浏览器备份。

服务支持 `CANVAS_PORT`（默认 8000）和 `CANVAS_DATA_DIR`（默认 canvas 目录），可用临时数据目录执行隔离测试。新增 `node canvas/test_runtime.cjs` 验证 PyTorch 遮罩操作、各 diagonal、Literal 标量优先、因果注意力上三角零值/行和/未来输入不变性、追踪与未追踪梯度一致、嵌套多输出与多调用，以及项目/代码/未保存修改的过期判定；`.venv/Scripts/python.exe canvas/test_runtime.py` 验证预览预算、非连续/空/标量张量、非有限数值、原 autograd、输入错误、部分 HTTP 追踪、结构化文件及草稿撤销 revision。子代理没有浏览器能力，真实拖拽、截图及可访问性由主会话另行验收。

## 环境

- Python 3.11.3，虚拟环境位于 `.venv/`，由 [uv](https://docs.astral.sh/uv/) 管理
- PyTorch 2.11.0+cu128（GPU 版），已验证可在 RTX 3050 Ti 上运行 CUDA

## 激活环境

Git Bash：

```bash
cd transformer-from-scratch
source .venv/Scripts/activate
```

PowerShell：

```powershell
cd transformer-from-scratch
.venv\Scripts\Activate.ps1
```

激活后直接用 `python` / `pip` 即可；或者不激活，每次用 `.venv/Scripts/python.exe`。

## 已安装的包

| 包 | 用途 |
|---|---|
| torch (CUDA) | 核心框架，手搓所有模块 |
| numpy | 数值计算、数据处理 |
| matplotlib | 画注意力热力图、loss 曲线 |
| tqdm | 训练进度条 |
| jupyter + ipykernel | 交互式学习、做笔记 |
| tensorboard | 训练过程可视化 |
| datasets + tokenizers | HuggingFace，下载语料（如 tiny shakespeare）、参考现成分词器 |

完整锁定版本见 `requirements.txt`。

## 常用命令（uv）

```bash
# 装新包
uv pip install --python .venv <package>

# 查看已装包
uv pip list --python .venv

# 按 requirements.txt 重建环境（换机器时）
uv venv --python 3.11 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt
```

注意：torch 必须从 PyTorch 官方源装才是 GPU 版（PyPI 上的 Windows 版只有 CPU）。

## 自检

```bash
python check_env.py
```

应输出 CUDA available: True 和显卡型号。

## Jupyter

```bash
# 把当前 venv 注册为 kernel 后启动
python -m ipykernel install --user --name transformer-from-scratch
jupyter lab   # 或 jupyter notebook
```
