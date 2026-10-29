"""validation.py（モデルのフィールド検証関数）のテスト。"""

from datetime import UTC

import pytest

from experiment_workbench.validation import (
    reject_duplicates,
    require_non_blank,
    utc_now,
    validate_relative_posix_path,
    validate_tags,
)


@pytest.mark.parametrize(
    "path", ["a", "datasets/x.parquet", "originals/trial 1/データ.csv", "a.b/c..d"]
)
def test_valid_relative_paths_are_accepted(path: str) -> None:
    """正規化済みの相対パスはそのまま返る。"""
    assert validate_relative_posix_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",  # 空
        "/abs",  # POSIX の絶対パス
        "C:/x",  # Windows のドライブ付き絶対パス
        "C:x",  # ドライブ相対パス
        "a\\b",  # バックスラッシュ区切り
        "../x",  # 親ディレクトリへの移動
        "a/../b",
        "./a",
        ".",
        "a//b",  # 空要素
        "a/",  # 末尾の区切り
    ],
)
def test_invalid_relative_paths_are_rejected(path: str) -> None:
    """プロジェクト外を指しうるパスや非正規形のパスは ValueError になる。"""
    with pytest.raises(ValueError, match=r"."):
        validate_relative_posix_path(path)


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_blank_strings_are_rejected(value: str) -> None:
    """空文字・空白のみは拒否される。"""
    with pytest.raises(ValueError, match="空"):
        require_non_blank(value)


def test_non_blank_string_is_kept_as_is() -> None:
    """前後の空白は勝手に除去しない（入力を隠して修正しない方針）。"""
    assert require_non_blank(" name ") == " name "


def test_reject_duplicates_keeps_order_without_duplicates() -> None:
    """重複がなければ入力と同じリスト（順序も同じ）が返る。"""
    assert reject_duplicates(["b", "a"]) == ["b", "a"]


def test_reject_duplicates_lists_duplicated_values() -> None:
    """重複があればエラーメッセージに重複要素が含まれる。"""
    with pytest.raises(ValueError, match="'a'"):
        reject_duplicates(["a", "b", "a"])


def test_validate_tags_rejects_blank_and_duplicate_tags() -> None:
    """タグは空白のみ・重複のどちらもエラーになり、正常なら入力がそのまま返る。"""
    assert validate_tags(["success", "slow"]) == ["success", "slow"]
    with pytest.raises(ValueError, match="空"):
        validate_tags(["success", ""])
    with pytest.raises(ValueError, match="重複"):
        validate_tags(["success", "success"])


def test_utc_now_is_timezone_aware_utc() -> None:
    """作成日時は UTC のタイムゾーン付きで返る。"""
    assert utc_now().tzinfo is UTC
