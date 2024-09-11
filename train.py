import xgboost as xgb
import pathlib
from sklearn.datasets import make_regression

n_samples = 16000
n_features = 32
n_trees = 512
max_depth = 8

model_dir = pathlib.Path("./model_repository/xgb_model/1")
model_dir.mkdir(exist_ok=True)

X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_features // 3,
    random_state=0,
)
X, y = X.astype("float32"), y.astype("float32")

clf = xgb.XGBRegressor(
    n_estimators=n_trees,
    objective="reg:squarederror",
    max_depth=max_depth,
    tree_method="hist",
    device="cpu",
    random_state=0,
    n_jobs=-1,
)
clf.fit(X, y, eval_set=[(X, y)])
clf.save_model(model_dir / "xgboost.json")
