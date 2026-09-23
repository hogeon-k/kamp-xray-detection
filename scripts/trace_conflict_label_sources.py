"""Trace conflict YOLO TXT bytes back to the supplied labeling-set TXT pool."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
METADATA = ROOT / "outputs" / "unique_512_review" / "metadata.csv"
CONFLICTS = ROOT / "outputs" / "unique_512_audit" / "label_conflicts.csv"
OFFICIAL_POOL = DATASET / "라벨링 6종 세트" / "labels"
OUT = ROOT / "outputs" / "label_provenance"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    baseline = {str(path): sha256(path) for path in (METADATA, CONFLICTS)}
    metadata = {row["unique_id"]: row for row in load(METADATA)}
    conflicts = load(CONFLICTS)
    assert len(metadata) == 512 and len(conflicts) == 23
    pool = sorted(OFFICIAL_POOL.glob("*.txt"))
    pool_hashes = {path: sha256(path) for path in pool}
    pool_by_hash = defaultdict(list)
    for path, digest in pool_hashes.items():
        pool_by_hash[digest].append(path)

    traces = []
    summaries = []
    zero_cases = []
    source_baseline = {}
    for conflict in conflicts:
        uid = conflict["unique_id"]
        meta = metadata[uid]
        paths = [Path(item) for item in json.loads(conflict["label_paths"])]
        xml_paths = [path for path in paths if path.suffix.lower() == ".xml"]
        yolo_paths = [path for path in paths if path.suffix.lower() == ".txt"]
        assert len(xml_paths) == 1 and yolo_paths
        xml_path = xml_paths[0]
        rep = Path(meta["representative_path"])
        official = OFFICIAL_POOL / f"{rep.stem}.txt"
        for path in [rep, xml_path, *yolo_paths]:
            if not path.is_file():
                raise FileNotFoundError(path)
            source_baseline[str(path)] = sha256(path)
        official_exists = official.is_file()
        official_hash = pool_hashes.get(official, "")
        official_bytes = official.read_bytes() if official_exists else None
        primary = next((path for path in yolo_paths if path == official), yolo_paths[0])
        label_rows = []
        for path in yolo_paths:
            digest = source_baseline[str(path)]
            byte_equal = official_bytes is not None and path.read_bytes() == official_bytes
            digest_equal = official_exists and digest == official_hash
            if byte_equal != digest_equal:
                raise AssertionError(f"Hash/content comparison disagrees for {path}")
            item = dict(unique_id=uid, representative_bmp_path=str(rep), xml_path=str(xml_path),
                        yolo_path=str(path), yolo_source_folder=str(path.parent),
                        yolo_role="PRIMARY_OFFICIAL_POOL" if path == official else "ADDITIONAL_COPY_OR_ALTERNATIVE",
                        official_same_stem_path=str(official), official_same_stem_exists=official_exists,
                        exact_same_stem_official_file=path == official,
                        same_content_as_official_txt=byte_equal,
                        sha256_equal_to_official_txt=digest_equal,
                        yolo_sha256=digest, official_txt_sha256=official_hash,
                        other_official_pool_hash_matches=json.dumps(
                            [str(p) for p in pool_by_hash[digest] if p != official], ensure_ascii=False),
                        external_organizer_origin_verified=False)
            traces.append(item)
            label_rows.append(item)
        primary_trace = next(row for row in label_rows if row["yolo_path"] == str(primary))
        summary = dict(unique_id=uid, representative_bmp_path=str(rep), xml_path=str(xml_path),
                       primary_yolo_path=str(primary), primary_yolo_source_folder=str(primary.parent),
                       official_txt_path=str(official), official_txt_exists=official_exists,
                       primary_is_official_pool_file=primary == official,
                       primary_content_equals_official=primary_trace["same_content_as_official_txt"],
                       primary_sha256_equals_official=primary_trace["sha256_equal_to_official_txt"],
                       primary_yolo_sha256=primary_trace["yolo_sha256"], official_txt_sha256=official_hash,
                       yolo_label_paths=json.dumps([str(path) for path in yolo_paths], ensure_ascii=False),
                       all_yolo_paths_equal_official=all(row["same_content_as_official_txt"] for row in label_rows),
                       xml_bbox_count=len(ET.parse(xml_path).getroot().findall("object")),
                       yolo_bbox_count=sum(bool(line.strip()) for line in primary.read_text(encoding="utf-8-sig").splitlines()),
                       external_organizer_origin_verified=False)
        summaries.append(summary)
        if summary["xml_bbox_count"] == 0 and summary["yolo_bbox_count"] > 0:
            zero_cases.append(summary)

    if len(zero_cases) != 14:
        raise AssertionError(f"Expected 14 XML=0/YOLO>0 cases, found {len(zero_cases)}")
    if any(not row["primary_is_official_pool_file"] or not row["primary_content_equals_official"]
           for row in zero_cases):
        raise AssertionError("A zero-XML case does not use the supplied label pool TXT")
    assert all(sha256(Path(path)) == digest for path, digest in source_baseline.items())
    assert all(sha256(path) == digest for path, digest in pool_hashes.items())
    assert all(sha256(Path(path)) == digest for path, digest in baseline.items())

    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / "all_yolo_source_traces.csv", traces, list(traces[0]))
    save(OUT / "conflict_23_source_summary.csv", summaries, list(summaries[0]))
    save(OUT / "xml_zero_yolo_positive_14.csv", zero_cases, list(summaries[0]))
    stats = dict(conflict_images=len(summaries), xml_zero_yolo_positive_images=len(zero_cases),
                 yolo_files_traced=len(traces), official_pool_txt_files=len(pool),
                 primary_matches_official_count=sum(row["primary_content_equals_official"] for row in summaries),
                 all_yolo_files_match_official_count=sum(row["same_content_as_official_txt"] for row in traces),
                 zero_xml_primary_matches_official_count=sum(row["primary_content_equals_official"] for row in zero_cases),
                 images_with_alternative_yolo=sum(len(json.loads(row["yolo_label_paths"])) > 1 for row in summaries),
                 external_organizer_origin_verified=False)
    (OUT / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 충돌 YOLO TXT 출처 추적\n\n"
        "이 결과는 기존 `metadata.csv`와 `label_conflicts.csv`의 23개 충돌 대상을 사용합니다. "
        "각 YOLO TXT의 경로·SHA-256·바이트를 `dataset/라벨링 6종 세트/labels`의 같은 파일명 TXT와 비교했습니다. "
        "이미지나 라벨을 수정하지 않았습니다.\n\n"
        "- `conflict_23_source_summary.csv`: 이미지별 대표 YOLO 출처와 공식 라벨 풀 일치 여부\n"
        "- `xml_zero_yolo_positive_14.csv`: XML=0, YOLO>0인 14장만 분리\n"
        "- `all_yolo_source_traces.csv`: 23장에 연결된 모든 YOLO TXT(대체 사본 포함)를 파일별로 추적\n\n"
        "여기서 '공식 라벨 풀'은 현재 프로젝트에 포함된 `라벨링 6종 세트/labels`를 뜻합니다. "
        "같은 파일임과 내용 일치는 검증했지만, 이 폴더 자체가 주최 측에서 배포되었다는 "
        "독립적인 배포 문서·manifest는 프로젝트에서 확인되지 않았습니다. 외부 제작·배포 주체는 미확인입니다.\n",
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
