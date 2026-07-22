# -*- coding: utf-8 -*-
"""云端训练入口 (P11, ARCH §2).

AutoDL A100: 下载全 A 股 2 年日线 → 因子计算 → 分池训练 →
因子发现 → ONNX 导出 + 特征快照 → 上传 Model Registry。
"""

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """云端训练主流程.

    Steps:
        1. download_full_a_share(): 全 A 股 2 年日线
        2. train_universe(): MAIN_BOARD / GROWTH_BOARDS 分池训练
        3. run_factor_discovery(): LGBM+SHAP+IC/ICIR 因子报告
        4. export_onnx(): 导出 ONNX + feature_snapshot.parquet
        5. upload_registry(): 上传模型与快照
    """
    raise NotImplementedError("P11 待建")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
