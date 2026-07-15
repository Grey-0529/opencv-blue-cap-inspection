from pathlib import Path

import cv2
import numpy as np


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "examples"
OUTPUT_DIR = PROJECT_ROOT / "output"

OK_PATH = INPUT_DIR / "sample_ok.jpg"
NG_PATH = INPUT_DIR / "sample_ng.jpg"



def inspect_cap(image_path: Path, sample_name: str) -> float:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    height, width = image.shape[:2]

    # 瓶盖区域 ROI
    # 如果框的位置不合适，后面再微调这四个比例
    x1 = int(width * 0.25)
    x2 = int(width * 0.50)
    y1 = int(height * 0.01)
    y2 = int(height * 0.16)

    cap_roi = image[y1:y2, x1:x2].copy()

    # 转换到 HSV
    cap_hsv = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2HSV)

    # 蓝色范围
    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])

    # 生成蓝色掩膜
    cap_blue_mask = cv2.inRange(
        cap_hsv,
        lower_blue,
        upper_blue,
    )

    # 创建 5×5 椭圆形卷积核
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    # 开运算：先腐蚀再膨胀，去掉零散白色噪点
    opened_mask = cv2.morphologyEx(
        cap_blue_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # 闭运算：先膨胀再腐蚀，填补蓝色区域内部的小黑洞
    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # 计算蓝色像素比例
    raw_blue_pixels = cv2.countNonZero(cap_blue_mask)
    cleaned_blue_pixels = cv2.countNonZero(cleaned_mask)

    total_pixels = cleaned_mask.shape[0] * cleaned_mask.shape[1]

    raw_blue_ratio = raw_blue_pixels / total_pixels
    blue_ratio = cleaned_blue_pixels / total_pixels

    print(f"{sample_name} 原始蓝色比例：{raw_blue_ratio:.2%}")
    print(f"{sample_name} 清理后蓝色比例：{blue_ratio:.2%}")

    # 查找清理后掩膜中的外部轮廓
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contour_result = cap_roi.copy()

    largest_contour = None
    largest_contour_area = 0.0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_area = cv2.contourArea(largest_contour)

        # 最大轮廓的外接矩形
        box_x, box_y, box_w, box_h = cv2.boundingRect(largest_contour)

        # 绿色：实际轮廓
        cv2.drawContours(
            contour_result,
            [largest_contour],
            -1,
            (0, 255, 0),
            2,
        )

        # 红色：外接矩形
        cv2.rectangle(
            contour_result,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (0, 0, 255),
            2,
        )

    # 最大连续蓝色区域占整个 ROI 的比例
    largest_contour_ratio = largest_contour_area / total_pixels

    print(f"{sample_name} 轮廓数量：{len(contours)}")
    print(f"{sample_name} 最大轮廓面积：{largest_contour_area:.2f}")
    print(f"{sample_name} 最大轮廓面积占比：{largest_contour_ratio:.2%}")

    BLUE_RATIO_THRESHOLD = 0.20
    CONTOUR_RATIO_THRESHOLD = 0.15

    blue_ratio_ok = blue_ratio >= BLUE_RATIO_THRESHOLD
    contour_ratio_ok = largest_contour_ratio >= CONTOUR_RATIO_THRESHOLD

    inspection_ok = blue_ratio_ok and contour_ratio_ok

    if inspection_ok:
        inspection_result = "OK"
        result_color = (0, 255, 0)
        failure_reason = "None"
    else:
        inspection_result = "NG"
        result_color = (0, 0, 255)

        if not blue_ratio_ok and not contour_ratio_ok:
            failure_reason = "Low blue ratio and small contour"
        elif not blue_ratio_ok:
            failure_reason = "Low blue ratio"
        else:
            failure_reason = "Small main contour"

    # 在原图上画出 ROI
    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        result_color,
        4,
    )

    cv2.putText(
        annotated,
        f"Result: {inspection_result}",
        (x1, min(y2 + 45, height - 100)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        result_color,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        f"Blue: {blue_ratio:.2%}",
        (x1, min(y2 + 85, height - 60)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        result_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        f"Contour: {largest_contour_ratio:.2%}",
        (x1, min(y2 + 125, height - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        result_color,
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        4,
    )

    cv2.putText(
        annotated,
        f"{sample_name} blue ratio: {blue_ratio:.2%}",
        (x1, min(y2 + 50, height - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 保存每张图的 ROI、Mask 和标注结果
    cv2.imwrite(
        str(OUTPUT_DIR / f"{sample_name}_roi.jpg"),
        cap_roi,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{sample_name}_mask.jpg"),
        cap_blue_mask,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{sample_name}_annotated.jpg"),
        annotated,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{sample_name}_contour.jpg"),
        contour_result,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{sample_name}_final_result.jpg"),
        annotated,
    )

    return {
        "blue_ratio": blue_ratio,
        "contour_ratio": largest_contour_ratio,
        "contour_count": len(contours),
        "result": inspection_result,
        "reason": failure_reason,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok_result = inspect_cap(OK_PATH, "ok")
    ng_result = inspect_cap(NG_PATH, "ng")

    print("\n===== 最终检测结果 =====")

    print(
        f"正常样本：{ok_result['result']} | "
        f"蓝色比例={ok_result['blue_ratio']:.2%} | "
        f"最大轮廓比例={ok_result['contour_ratio']:.2%}"
    )

    print(
        f"异常样本：{ng_result['result']} | "
        f"蓝色比例={ng_result['blue_ratio']:.2%} | "
        f"最大轮廓比例={ng_result['contour_ratio']:.2%} | "
        f"原因={ng_result['reason']}"
    )


if __name__ == "__main__":
    main()
