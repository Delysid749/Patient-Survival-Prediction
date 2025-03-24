from src.config.path_config import DATA_FILE, MODEL_DIR
import pandas as pd
import cloudpickle
from src.model.Ensemble import predict_new_patients
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

if __name__ == '__main__':
    #  使用 cloudpickle 加载模型
    with open(MODEL_DIR / "base_models.pkl", "rb") as f:
        base_models = cloudpickle.load(f)

    with open(MODEL_DIR / "meta_model.pkl", "rb") as f:
        meta_model = cloudpickle.load(f)

    #  加载数据，选取第91611行进行预测
    df = pd.read_csv(DATA_FILE)
    single_row = df.iloc[[91611]].drop(columns=['hospital_death'])  # 保证结构一致
    print(single_row["encounter_id"])
    #  执行预测
    pred, prob = predict_new_patients(single_row, base_models, meta_model)

    #  输出预测结果
    print("预测结果：", "死亡" if pred[0] == 1 else "存活")
    print("预测概率：", round(prob[0], 4))
