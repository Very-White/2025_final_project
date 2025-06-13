"""
check_translations.py  ―  校验翻译结果 JSON 文件格式

Usage:
    python check_translations.py translations.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

REQUIRED_KEYS = {"src", "ref", "hyp", "hyp_id"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="检查翻译文件是否满足 src/ref/hyp/hyp_id 四字段格式"
    )
    p.add_argument("file", metavar="FILE", type=str, help="译文 JSON 文件路径")
    return p.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] JSON 解析失败: {e}")
    if not isinstance(data, list):
        sys.exit("[ERROR] 顶层结构必须是列表(list)。")
    return data


def validate_records(records: List[Dict[str, Any]]) -> None:
    seen_ids = set()
    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            sys.exit(f"[ERROR] 第 {idx} 条记录应为对象(dict)，实际为: {type(rec).__name__}")

        missing = REQUIRED_KEYS - rec.keys()
        if missing:
            sys.exit(f"[ERROR] 第 {idx} 条记录缺少字段: {', '.join(sorted(missing))}")


    print(f"文件格式无误，共 {len(records)} 条记录。")


def main() -> None:
    args = parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        sys.exit(f"[ERROR] 文件不存在: {file_path}")

    records = load_json(file_path)
    validate_records(records)


if __name__ == "__main__":
    main()
