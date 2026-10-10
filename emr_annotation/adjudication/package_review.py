"""Package an existing offline adjudication page for controlled distribution."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(review_dir, output):
    root = Path(__file__).resolve().parents[2]
    review_dir = review_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    files = {
        'index.html': review_dir / 'review.html',
        '核查报告.md': review_dir / '核查报告.md',
        '裁决模板.json': review_dir / '裁决模板.json',
        '标签核查签署模板.json': review_dir / '标签核查签署模板.json',
        '验证记录_v2.json': review_dir / '验证记录_v2.json',
        '裁决方案.md': root / 'documentation/double_annotation_adjudication/double-annotation-adjudication.md',
        '工作台功能说明.md': root / 'documentation/double_annotation_adjudication/double-annotation-workbench.md',
        '使用与发布说明.md': root / 'documentation/project_delivery/double-annotation-release.md',
    }
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    if '__PAYLOAD__' in files['index.html'].read_text('utf-8'):
        raise ValueError('Page is an unbuilt template')
    site = output / 'website'
    operator = output / 'operator'
    site.mkdir(parents=True)
    operator.mkdir()
    shutil.copy2(files['index.html'], site / 'index.html')
    for name, source in files.items():
        shutil.copy2(source, operator / name)
    report = operator / '核查报告.md'
    report.write_text(re.sub(
        r'(\[[^\]]*\]\()(?:(?!\)).)*double-annotation-adjudication\.md(\))',
        r'\1裁决方案.md\2', report.read_text('utf-8')), 'utf-8')
    manual = operator / '工作台功能说明.md'
    manual.write_text(manual.read_text('utf-8').replace(
        '../project_delivery/double-annotation-release.md', '使用与发布说明.md'), 'utf-8')
    manifest = {
        'release': 'adjudication-workbench-v2',
        'distribution': 'controlled_access_only_contains_emr_fragments',
        'source_review_directory': str(review_dir),
        'source_page_sha256': sha256(files['index.html']),
        'page_copies_byte_identical': sha256(files['index.html']) == sha256(site / 'index.html') == sha256(operator / 'index.html'),
        'browser_drafts_included': False,
        'files': [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in sorted(output.rglob('*')) if p.is_file()],
    }
    (operator / '发布文件校验清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    for folder, archive in [(site, '工作台网页_内网发布.zip'), (operator, '裁决员协作包.zip')]:
        with ZipFile(output / archive, 'w', ZIP_DEFLATED) as zip_file:
            for path in sorted(folder.iterdir()):
                zip_file.write(path, path.name)
    manifest['archives'] = [{'path': name, 'sha256': sha256(output / name)}
                            for name in ['工作台网页_内网发布.zip', '裁决员协作包.zip']]
    (output / '发布清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    (output / '发布选择说明.md').write_text(
        '# 发布选择\n\n'
        '- 网站上传：选择`website/index.html`；如果平台接收ZIP，选择`工作台网页_内网发布.zip`。\n'
        '- 发给裁决员：选择`裁决员协作包.zip`，解压后打开`index.html`。\n'
        '- 不用上传原始导出、audit JSON、Python脚本或源码HTML模板。\n\n'
        '本包包含病历片段，只用于受控内网或有访问权限的平台。页面不提供登录或多人实时同步。'
        '个人草稿需从现有浏览器导出JSON再导入发布地址；发布包不含这些草稿。\n', 'utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = package(args.review_dir, args.output)
    print(json.dumps({'page_copies_byte_identical': result['page_copies_byte_identical'],
                      'archives': [entry['path'] for entry in result['archives']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
