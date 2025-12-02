import os
import json
import torch
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from utils_train import TabDiffDataset
import src

print("\n================= TEST: Real Adult Data + Causal Noise =================\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device =", device)

# ----------------------------------------------------------------------------
# 1. 加载 info.json
# ----------------------------------------------------------------------------
info_path = "data/adult/info.json"
with open(info_path, "r") as f:
    info = json.load(f)

idx_name_mapping = info["idx_name_mapping"]
num_col_idx = info["num_col_idx"]
cat_col_idx = info["cat_col_idx"]

print("\nColumn mapping loaded.")
print("Numerical columns:", num_col_idx)
print("Categorical columns:", cat_col_idx)

# ----------------------------------------------------------------------------
# 2. 加载 DataSet
# ----------------------------------------------------------------------------
train_data = TabDiffDataset(
    "adult",
    "data/adult",
    info,
    y_only=False,
    isTrain=True,
    dequant_dist="uniform",
    int_dequant_factor=1.0
)

# 取 4 条真实样本
batch_np = train_data.X[:4]     # numpy array (4, 15)
batch = torch.tensor(batch_np, dtype=torch.float32).to(device)

print("\nReal batch loaded, shape =", batch.shape)

# ----------------------------------------------------------------------------
# 3. 加载作者的 config（关键点）
# ----------------------------------------------------------------------------
curr_dir = os.path.dirname(__file__)
config_path = f"{curr_dir}/tabdiff/configs/tabdiff_configs.toml"
raw_config = src.load_config(config_path)

# 注入必要字段
raw_config["unimodmlp_params"]["d_numerical"] = len(num_col_idx)
raw_config["unimodmlp_params"]["categories"] = (train_data.categories + 1).tolist()

print("\nMLP Config loaded:")
print(raw_config["unimodmlp_params"])

# ----------------------------------------------------------------------------
# 4. 构建 UniModMLP + Model
# ----------------------------------------------------------------------------
backbone = UniModMLP(**raw_config["unimodmlp_params"])
model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
model.to(device)

# ----------------------------------------------------------------------------
# 5. 构建 Diffusion + 噪声分层参数
# ----------------------------------------------------------------------------
d_numerical = len(num_col_idx)
num_classes = train_data.categories

diffusion = UnifiedCtimeDiffusion(
    num_classes=num_classes,
    num_numerical_features=d_numerical,
    denoise_fn=model,
    y_only_model=None,
    **raw_config["diffusion_params"],
    device=device,

    idx_name_mapping=info["idx_name_mapping"],
    num_col_idx=info["num_col_idx"],
    cat_col_idx=info["cat_col_idx"],
    target_col_idx=info["target_col_idx"],
    causal_layers_path="data/adult/layers.json",
)


diffusion.to(device)
diffusion.train()

# ====================================================================
#  ★★★ 打印 UnifiedCtimeDiffusion 内部真实映射（你要的部分） ★★★
# ====================================================================
print("\n================= CHECK: REAL SCALE MAPPING =================\n")

with open("data/adult/layers.json", "r") as f:
    layer_cfg = json.load(f)

num_layers = layer_cfg["num_layers"]
layer_scales = torch.linspace(0.6, 1.2, steps=num_layers)

# 构造 {特征名 → (层名, 噪声scale)} 映射
layer_table = {}   # feat_name → (layer_name, scale)
for i in range(num_layers):
    for feat in layer_cfg[f"layer{i}"]:
        layer_table[feat] = (f"layer{i}", float(layer_scales[i]))


# ========== 打印 numerical ==========
print("NUMERIC FEATURES:\n")
for local_idx, col_idx in enumerate(num_col_idx):
    feat = idx_name_mapping[str(col_idx)]
    layer_name, expected_scale = layer_table.get(feat, ("(not found)", 1.0))
    actual_scale = float(diffusion.num_layer_scale[local_idx])
    print(f"[num] col={col_idx:2d}  feat={feat:20s}  layer={layer_name:10s}  scale(expected)={expected_scale:.2f}  scale(used)={actual_scale:.2f}")


# ========== 打印 categorical ==========
print("\nCATEGORICAL FEATURES:\n")
for local_idx, col_idx in enumerate(cat_col_idx):
    feat = idx_name_mapping[str(col_idx)]
    layer_name, expected_scale = layer_table.get(feat, ("(not found)", 1.0))
    actual_scale = float(diffusion.cat_layer_scale[local_idx])
    print(f"[cat] col={col_idx:2d}  feat={feat:20s}  layer={layer_name:10s}  scale(expected)={expected_scale:.2f}  scale(used)={actual_scale:.2f}")

print("\n================= END SCALE CHECK =================\n")


# ----------------------------------------------------------------------------
# 6. 执行一次 forward，测试是否跑通
# ----------------------------------------------------------------------------
try:
    print("\nRunning mixed_loss...")
    d_loss, c_loss = diffusion.mixed_loss(batch)
    print("\nSUCCESS! d_loss =", d_loss.item(), ", c_loss =", c_loss.item())

except Exception as e:
    print("\nERROR during forward:", e)

print("\n================= END TEST =================\n")
