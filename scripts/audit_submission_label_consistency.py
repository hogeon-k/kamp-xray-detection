"""Verify the selected 512 labels against project sources without changing inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from audit_missing_12_labels import box_equivalent
from create_clean_512_candidate import validate_label
from create_label_review import parse_xml, parse_yolo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "submission_label_audit"


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    manifest = read_csv("outputs/yolo_512_telea/manifest.csv")
    metadata = {r["unique_id"]: r for r in read_csv("outputs/clean_512/metadata.csv")}
    provenance = {r["unique_id"]: r for r in read_csv("outputs/clean_512/provenance.csv")}
    split_manifest = {r["unique_id"]: r for r in read_csv("outputs/clean_512_split/split_manifest.csv")}
    legacy = read_csv("outputs/label_provenance/conflict_23_source_summary.csv")
    traces = read_csv("outputs/label_provenance/all_yolo_source_traces.csv")
    historical_ids = {r["unique_id"] for r in legacy}
    assert len(manifest) == len(metadata) == len(provenance) == 512
    assert len(historical_ids) == 23
    assert {r["unique_id"] for r in manifest} == set(metadata) == set(provenance)

    protected: dict[Path, str] = {}

    def check_file(path: Path) -> str:
        path = path.resolve()
        assert path.is_relative_to(ROOT), path
        assert path.is_file(), path
        current = digest(path)
        if path in protected:
            assert protected[path] == current, path
        protected[path] = current
        return current

    rows = []
    for record in manifest:
        uid, filename, split = record["unique_id"], record["filename"], record["split"]
        stem = Path(filename).stem
        meta, prov = metadata[uid], provenance[uid]
        clean_label = ROOT / "outputs/clean_512/labels" / f"{stem}.txt"
        selected_source = Path(prov["final_label_original_path"])
        telea_label = ROOT / "outputs/yolo_512_telea/labels" / split / f"{stem}.txt"
        label_sha = check_file(clean_label)
        source_sha = check_file(selected_source)
        telea_sha = check_file(telea_label)
        split_sha = split_manifest[uid]["label_sha256"]
        assert split_manifest[uid]["split"] == split
        boxes, errors = validate_label(clean_label, int(meta["width"]), int(meta["height"]), uid)
        source_match = source_sha == label_sha and selected_source.read_bytes() == clean_label.read_bytes()
        image_sha = check_file(ROOT / "outputs/clean_512/images" / filename)
        original_image_sha = check_file(Path(meta["representative_original_path"]))
        restored_sha = check_file(ROOT / "outputs/yolo_512_telea/images" / split / record["restored_filename"])
        canonical = ROOT / "dataset/라벨링 6종 세트/labels" / f"{stem}.txt"
        fallback_match, fallback_xml_match = "", ""
        canonical_match: bool | str = ""
        if record["label_source_type"] == "CANONICAL_POOL":
            canonical_match = check_file(canonical) == label_sha and canonical.read_bytes() == clean_label.read_bytes()
            assert selected_source.resolve() == canonical.resolve()
        else:
            assert record["label_source_type"] == "VERIFIED_FALLBACK" and not canonical.exists()
            open_label = ROOT / "dataset/OpenLabeling-master/main/output/YOLO_darknet" / f"{stem}.txt"
            test_label = ROOT / "dataset/test1/yolov3/labels" / f"{stem}.txt"
            xml_label = ROOT / "dataset/OpenLabeling-master/main/output/PASCAL_VOC" / f"{stem}.xml"
            fallback_match = check_file(open_label) == check_file(test_label) == label_sha
            fallback_match = fallback_match and open_label.read_bytes() == test_label.read_bytes() == clean_label.read_bytes()
            check_file(xml_label)
            fallback_xml_match = box_equivalent(
                parse_xml(xml_label), parse_yolo(clean_label, int(meta["width"]), int(meta["height"])), 1.25
            )
        assert source_match and telea_sha == split_sha == label_sha
        assert label_sha == meta["sha256_label"] == record["label_sha256"] == prov["final_label_sha256"]
        assert image_sha == original_image_sha == meta["sha256_image"] == record["image_sha256"]
        assert restored_sha == record["restored_sha256"]
        assert not errors and len(boxes) == int(record["bbox_count"])
        assert canonical_match is True or (fallback_match is True and fallback_xml_match is True)
        rows.append(dict(
            unique_id=uid, filename=filename, split=split, bbox_count=len(boxes),
            label_source_type=record["label_source_type"],
            historical_xml_yolo_conflict=uid in historical_ids,
            selected_source_match=source_match, canonical_final_match=canonical_match,
            fallback_txt_copies_match=fallback_match, fallback_xml_bbox_class_match=fallback_xml_match,
            split_manifest_label_match=split_sha == label_sha, telea_label_match=telea_sha == label_sha,
            recorded_label_hash_match=True, original_clean_image_hash_match=True,
            restored_image_hash_match=True, valid_yolo_format=True,
            selected_source_sha256=source_sha, final_label_sha256=label_sha,
            legacy_review_status=meta["review_status"], current_verification="SELECTED_SOURCE_MATCH",
        ))

    trace_rows = []
    for row in traces:
        source, canonical = Path(row["yolo_path"]), Path(row["official_same_stem_path"])
        source_sha, canonical_sha = check_file(source), check_file(canonical)
        equal = source_sha == canonical_sha and source.read_bytes() == canonical.read_bytes()
        trace_rows.append(dict(unique_id=row["unique_id"], source_path=source.relative_to(ROOT).as_posix(),
                               canonical_path=canonical.relative_to(ROOT).as_posix(),
                               matches_canonical=equal, source_sha256=source_sha,
                               canonical_sha256=canonical_sha,
                               selected_in_final=source.resolve() == Path(provenance[row["unique_id"]]["final_label_original_path"]).resolve()))

    historical = [row for row in rows if row["historical_xml_yolo_conflict"]]
    assert len(historical) == 23 and all(row["canonical_final_match"] for row in historical)
    assert all(digest(path) == value for path, value in protected.items())
    summary = dict(
        status="passed", audit_time_local=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        images=512, bbox=sum(r["bbox_count"] for r in rows),
        selected_source_matches=sum(r["selected_source_match"] for r in rows),
        selected_source_mismatches=sum(not r["selected_source_match"] for r in rows),
        canonical_verified_images=sum(r["canonical_final_match"] is True for r in rows),
        fallback_txt_xml_verified_images=sum(r["fallback_xml_bbox_class_match"] is True for r in rows),
        historical_xml_yolo_different_images=23,
        historical_23_selected_canonical_matches=23,
        historical_23_selected_canonical_mismatches=0,
        historical_xml_zero_yolo_positive_images=sum(int(r["xml_bbox_count"]) == 0 for r in legacy),
        alternate_yolo_trace_files=len(trace_rows),
        alternate_yolo_matches_canonical=sum(r["matches_canonical"] for r in trace_rows),
        alternate_yolo_differences=sum(not r["matches_canonical"] for r in trace_rows),
        alternate_difference_image_ids=sorted({r["unique_id"] for r in trace_rows if not r["matches_canonical"]}),
        unselected_different_yolo_files=sum(not r["matches_canonical"] and not r["selected_in_final"] for r in trace_rows),
        legacy_review_status_counts=dict(Counter(r["legacy_review_status"] for r in rows)),
        legacy_review_status_meaning="Initialization values from dataset generation; separate from current source consistency verification",
        format_errors=0, split_label_mismatches=0, telea_label_mismatches=0,
        input_files_verified_unchanged=len(protected),
        raw_labels_modified=False, training_executed=False, inference_executed=False,
        verification_scope="Selected-label source/content/hash/format consistency, not expert semantic approval or equality of all legacy XML/alternative labels",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for name, values in [("selected_512_label_check.csv", rows), ("historical_23_selected_label_check.csv", historical),
                         ("alternate_yolo_trace_check.csv", trace_rows)]:
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
