"""Build and serve a local, source-backed human adjudication workbench.

No models or external services are used. Original exports and frozen references
are immutable; human drafts and their revisions are stored in a separate run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
from urllib.parse import urlparse, parse_qs

from emr_annotation.annotation.schema import load_double_annotation_schema
from emr_annotation.annotation_analysis.double_annotations import parse_annotation
from emr_annotation.training_data.schema_multitask import relation_type_allowed


FIELD_NAMES = {'finding_context': '存在状态', 'temporality': '时间属性',
               'clinical_course': '病程变化', 'conditional': '条件性',
               'hypothetical': '假设性', 'experiencer': '主体',
               'test_result': '检测结果', 'specimen_type': '标本类型',
               'detection_method': '检测方式', 'case_decision': '病例分类'}
POINT_STATUSES = {'accepted', 'modified', 'deleted', 'pending'}


@contextmanager
def database(path):
    # sqlite3's own context manager commits/rolls back but does not close files.
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            yield connection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), 'utf-8')


def valid_span(start, end, text):
    return type(start) is int and type(end) is int and 0 <= start < end <= len(text)


def group_case(sample, points, annotations):
    """Group overlapping selections; every ledger point is assigned exactly once."""
    text = sample['text']
    entities = {e['id']: e for e in sample['entities']}
    ranges = [(e['start'], e['end']) for e in sample['entities']]
    point_ranges = {}
    for point in points:
        raw = point.get('original') or {}
        start, end = raw.get('start'), raw.get('end')
        final = point.get('final')
        if point['kind'] == 'entity_added' and isinstance(final, dict):
            start, end = final.get('start'), final.get('end')
        elif point['kind'] == 'attribute_final' and isinstance(final, dict):
            ent = entities.get(final.get('entity_id'), {})
            start, end = ent.get('start'), ent.get('end')
        if valid_span(start, end, text):
            point_ranges[point['id']] = (start, end)
            ranges.append((start, end))
    merged = []
    for start, end in sorted(set(ranges)):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    groups = [{'id': f'span_{s}_{e}', 'start': s, 'end': e, 'point_ids': [],
               'entity_ids': [x['id'] for x in sample['entities']
                              if x['start'] < e and x['end'] > s]}
              for s, e in merged]
    special = {'case': {'id': 'case', 'point_ids': [], 'entity_ids': []},
               'relations': {'id': 'relations', 'point_ids': [], 'entity_ids': []},
               'other': {'id': 'other', 'point_ids': [], 'entity_ids': []}}
    for point in points:
        raw = point.get('original') or {}
        if point['kind'] in ('relation', 'relation_final'):
            group = special['relations']
        elif raw.get('from_name') == 'case_decision':
            group = special['case']
        elif point['id'] in point_ranges:
            s, e = point_ranges[point['id']]
            group = next(g for g in groups if g['start'] <= s and e <= g['end'])
        else:
            group = special['other']
        group['point_ids'].append(point['id'])
    allgroups = [special['case'], *groups, special['relations']]
    if special['other']['point_ids']:
        allgroups.append(special['other'])
    assigned = [p for group in allgroups for p in group['point_ids']]
    if Counter(assigned) != Counter(p['id'] for p in points):
        raise ValueError('Point grouping lost or duplicated source points')
    return allgroups


def build(source_dir, schema_path, output_dir, sources_path=None, findings_path=None):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    sources_path = Path(sources_path or source_dir.parent / 'sources.json')
    schema = load_double_annotation_schema(schema_path)
    samples = read(source_dir / 'proposals.json')
    ledger = read(source_dir / 'point_decisions.json')
    if len({s['task_id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate task IDs')
    if len({p['id'] for p in ledger}) != len(ledger):
        raise ValueError('Duplicate ledger IDs')
    if any(p['task_id'] not in {s['task_id'] for s in samples} for p in ledger):
        raise ValueError('Unknown ledger task')
    inputs = {str(Path(p).resolve()): sha(p) for p in
              (source_dir/'proposals.json', source_dir/'point_decisions.json', schema_path, sources_path)}
    tasks = defaultdict(list)
    for source in read(sources_path):
        path = Path(source['path'])
        if sha(path) != source['sha256']:
            raise ValueError('Changed original export: ' + str(path))
        inputs[str(path.resolve())] = sha(path)
        for task in read(path):
            tasks[(int(source['project']), task['id'])].append((task, path.name))
    findings = []
    if findings_path:
        inputs[str(Path(findings_path).resolve())] = sha(findings_path)
        info = read(findings_path)
        if info.get('new_finding'):
            findings.append(info['new_finding'])
    bycase = defaultdict(list)
    for point in ledger:
        bycase[point['task_id']].append(point)
    bundle_id = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'cases').mkdir()
    summaries = []
    original_count = 0
    for sample in samples:
        tid = sample['task_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', tid):
            raise ValueError('Unsafe task ID')
        points = bycase[tid]
        expected_ids = {p['source_annotation_id'] for p in points if p.get('source_annotation_id') is not None}
        matches = [(t, name) for t, name in tasks[(sample['project'], sample['source_task_id'])]
                   if t['data']['text'] == sample['text'] and
                   {a['id'] for a in t.get('annotations', [])} == expected_ids]
        if not matches:
            raise ValueError('Cannot match original annotation pair: ' + tid)
        task, filename = matches[0]
        annotations = []
        for number, ann in enumerate(sorted(task['annotations'], key=lambda a: a['id'])):
            parsed = parse_annotation(task, ann, schema, 'a'+str(ann['id']), tid)
            inventory_ids = {p['ref'] for p in parsed['inventory']}
            traced_ids = {p['id'] for p in points if p.get('source_annotation_id') == ann['id']}
            if inventory_ids != traced_ids:
                raise ValueError('Original results not fully traced: ' + tid)
            traced = {p['id']: p['original'] for p in points if p.get('source_annotation_id') == ann['id']}
            if any(row != traced[row['ref']] for row in parsed['inventory']):
                raise ValueError('Ledger differs from original export: ' + tid)
            original_count += len(inventory_ids)
            parsed['display_name'] = '标注'+chr(65+number)
            annotations.append(parsed)
        groups = group_case(sample, points, annotations)
        alerts = [f for f in findings if f['task_id'] == tid]
        case = {'sample': sample, 'points': points, 'groups': groups,
                'annotations': annotations, 'alerts': alerts, 'source_file': filename}
        write(output_dir/'cases'/f'{tid}.json', case)
        summaries.append({'task_id': tid, 'project': sample['project'],
                          'case_decision': sample['case_decision'],
                          'case_reason': sample['case_reason'], 'characters': len(sample['text']),
                          'entities': len(sample['entities']), 'relations': len(sample['relations']),
                          'points': len(points), 'groups': len(groups),
                          'pending': sum(p['decision'] == '待复核' for p in points),
                          'alerts': len(alerts), 'ner_supervised': sample['ner_supervised']})
    index = {'bundle_id': bundle_id, 'source_dir': str(source_dir.resolve()),
             'schema': schema, 'field_names': FIELD_NAMES, 'cases': summaries,
             'counts': {'cases': len(samples), 'original_results': original_count,
                        'points': len(ledger), 'groups': sum(x['groups'] for x in summaries),
                        'pending': sum(x['pending'] for x in summaries)},
             'reference_status': 'ai_sample_not_expert_gold'}
    write(output_dir/'index.json', index)
    template = Path(__file__).parent/'templates/human_review.html'
    (output_dir/'review.html').write_bytes(template.read_bytes())
    manifest = {'stage': 'adjudication', 'run_id': output_dir.name,
                'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
                'bundle_id': bundle_id, 'counts': index['counts'],
                'schema_sha256': sha(schema_path), 'source_overwritten': False,
                'medical_expert_acceptance': False, 'uses_model_or_network': False,
                'verification': {'all_source_results_traced': True,
                                 'all_ledger_points_assigned_once': True},
                'code_files': {str(p.resolve()): sha(p) for p in (Path(__file__), template)},
                'outputs': {str(p.relative_to(output_dir)): sha(p)
                            for p in output_dir.rglob('*') if p.is_file()}}
    try:
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        manifest['working_tree'] = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    except subprocess.CalledProcessError:
        manifest['git_commit'] = None
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('Input changed during generation')
    write(output_dir/'build_manifest.json', manifest)
    return index


def initial_document(case):
    sample = case['sample']
    return {'candidate': {k: deepcopy(sample[k]) for k in
                          ('entities', 'relations', 'case_decision', 'case_reason')},
            'point_reviews': {}, 'reviewer': '', 'case_note': '',
            'case_status': 'unreviewed', 'alerts_reviewed': False,
            'new_entities_reviewed': False}


def validate_document(document, case, schema):
    if not isinstance(document, dict):
        raise ValueError('Invalid review document')
    candidate = document.get('candidate', {})
    if not isinstance(candidate, dict) or set(candidate) != {'entities', 'relations', 'case_decision', 'case_reason'}:
        raise ValueError('裁决只能修改实体、关系、病例分类和理由；原文及患者归属不可修改')
    text = case['sample']['text']
    if candidate.get('case_decision') not in schema['fields']['case_decision']['values']:
        raise ValueError('病例分类不合法')
    if not isinstance(candidate.get('case_reason'), str):
        raise ValueError('病例理由必须是文字')
    entities = candidate.get('entities')
    relations = candidate.get('relations')
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError('实体和关系必须是列表')
    ids, keys = {}, set()
    for ent in entities:
        if not isinstance(ent, dict) or not isinstance(ent.get('id'), str) or not ent['id']:
            raise ValueError('实体ID不合法')
        if ent['id'] in ids:
            raise ValueError('实体ID重复')
        if not valid_span(ent.get('start'), ent.get('end'), text):
            raise ValueError('实体起止位置不合法')
        if ent.get('text') != text[ent['start']:ent['end']]:
            raise ValueError('实体文字与原文不一致')
        if ent.get('label') not in schema['labels']:
            raise ValueError('实体标签不合法')
        key = (ent['start'], ent['end'], ent['label'])
        if key in keys:
            raise ValueError('同跨度同标签实体重复')
        keys.add(key)
        ids[ent['id']] = ent
        attrs = ent.get('attributes', {})
        if not isinstance(attrs, dict):
            raise ValueError('属性格式不合法')
        for field, value in attrs.items():
            rule = schema['fields'].get(field)
            if not rule or not rule['per_region'] or ent['label'] not in rule['scope'] or value not in rule['values']:
                raise ValueError('属性值或适用范围不合法：'+field)
    relation_keys = set()
    for rel in relations:
        if (not isinstance(rel, dict) or rel.get('from_id') not in ids or rel.get('to_id') not in ids or
                rel['from_id'] == rel['to_id'] or rel.get('direction', 'right') != 'right' or
                not relation_type_allowed(ids[rel['from_id']]['label'], ids[rel['to_id']]['label'], rel.get('type'), schema)):
            raise ValueError('关系类型、方向或端点不合法')
        key = (rel['from_id'], rel['to_id'], rel['type'])
        if key in relation_keys:
            raise ValueError('关系重复')
        relation_keys.add(key)
    point_ids = {p['id'] for p in case['points']}
    reviews = document.get('point_reviews')
    if not isinstance(reviews, dict) or not set(reviews) <= point_ids:
        raise ValueError('审查点号不属于本例')
    for review in reviews.values():
        if not isinstance(review, dict) or review.get('status') not in POINT_STATUSES or not isinstance(review.get('note', ''), str):
            raise ValueError('逐点审查状态不合法')
    if document.get('case_status') not in ('unreviewed', 'reviewing', 'approved'):
        raise ValueError('病例审查状态不合法')
    for field in ('reviewer', 'case_note'):
        if not isinstance(document.get(field, ''), str):
            raise ValueError('审查人和意见必须是文字')
    if document.get('case_status') == 'approved':
        if not document.get('reviewer', '').strip():
            raise ValueError('确认全例时需填写审查人')
        if set(reviews) != point_ids or any(v['status'] == 'pending' for v in reviews.values()):
            raise ValueError('仍有未审查或暂缓的点，不能确认全例')
        if case['alerts'] and not document.get('alerts_reviewed'):
            raise ValueError('请先确认已处理已知裁决问题')
        if not document.get('new_entities_reviewed'):
            raise ValueError('请确认已核查全文遗漏及新增实体')
        for ent in entities:
            for field, rule in schema['fields'].items():
                if rule['per_region'] and rule['required'] and ent['label'] in rule['scope'] and field not in ent.get('attributes', {}):
                    raise ValueError('缺少必填属性：'+field)
    return document


class RevisionConflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root)
        self.index = read(self.root/'index.json')
        self.case_ids = {c['task_id'] for c in self.index['cases']}
        self.lock = threading.RLock()
        self.db = self.root/'review_state.sqlite3'
        with database(self.db) as con:
            con.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE IF NOT EXISTS drafts(task_id TEXT PRIMARY KEY,revision INTEGER,document TEXT,updated_at TEXT);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT,revision INTEGER,document TEXT,created_at TEXT);''')
            old = con.execute("SELECT value FROM meta WHERE key='bundle_id'").fetchone()
            if old and old[0] != self.index['bundle_id']:
                raise ValueError('Review database belongs to another source bundle')
            con.execute("INSERT OR IGNORE INTO meta VALUES('bundle_id',?)", (self.index['bundle_id'],))

    def case(self, tid):
        if tid not in self.case_ids:
            raise ValueError('Unknown case')
        return read(self.root/'cases'/f'{tid}.json')

    def get(self, tid):
        case = self.case(tid)
        with self.lock, database(self.db) as con:
            row = con.execute('SELECT revision,document,updated_at FROM drafts WHERE task_id=?', (tid,)).fetchone()
        return {'case': case, 'revision': row[0] if row else 0,
                'document': json.loads(row[1]) if row else initial_document(case),
                'updated_at': row[2] if row else None}

    def save(self, tid, expected_revision, document):
        case = self.case(tid)
        validate_document(document, case, self.index['schema'])
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, database(self.db) as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT revision FROM drafts WHERE task_id=?', (tid,)).fetchone()
            current = row[0] if row else 0
            if type(expected_revision) is not int or current != expected_revision:
                raise RevisionConflict('另一窗口已保存此病例，请重新加载后继续，当前编辑未覆盖服务器')
            revision = current+1
            encoded = json.dumps(document, ensure_ascii=False)
            con.execute('INSERT OR REPLACE INTO drafts VALUES(?,?,?,?)', (tid, revision, encoded, now))
            con.execute('INSERT INTO events(task_id,revision,document,created_at) VALUES(?,?,?,?)',
                        (tid, revision, encoded, now))
        return {'revision': revision, 'updated_at': now}

    def summary(self):
        with self.lock, database(self.db) as con:
            rows = con.execute('SELECT task_id,revision,document,updated_at FROM drafts').fetchall()
        return {tid: {'revision': rev, 'status': doc['case_status'],
                      'reviewer': doc.get('reviewer', ''),
                      'reviewed_points': sum(r['status'] != 'pending' for r in doc['point_reviews'].values()),
                      'updated_at': at} for tid, rev, encoded, at in rows for doc in [json.loads(encoded)]}

    def approve(self, tid, expected_revision):
        """Confirm the saved draft without changing it when completion fails."""
        with self.lock:
            current = self.get(tid)
            if type(expected_revision) is not int or current['revision'] != expected_revision:
                raise RevisionConflict('另一窗口已保存此病例，请重新加载后继续，当前编辑未覆盖服务器')
            document = deepcopy(current['document'])
            document['case_status'] = 'approved'
            result = self.save(tid, expected_revision, document)
            return {**result, 'case_status': 'approved'}

    def history(self, tid):
        self.case(tid)
        with self.lock, database(self.db) as con:
            rows = con.execute('SELECT revision,document,created_at FROM events WHERE task_id=? ORDER BY revision DESC LIMIT 100', (tid,)).fetchall()
        return [{'revision': rev, 'document': json.loads(encoded), 'created_at': at} for rev, encoded, at in rows]

    def export(self, approved_only=False):
        with self.lock, database(self.db) as con:
            rows = con.execute('SELECT task_id,revision,document,updated_at FROM drafts').fetchall()
        records = []
        for tid, rev, encoded, at in rows:
            doc = json.loads(encoded)
            if approved_only and doc['case_status'] != 'approved':
                continue
            case = self.case(tid)
            validate_document(doc, case, self.index['schema'])
            records.append({'task_id': tid, 'text_sha256': case['sample']['text_sha256'],
                            'revision': rev, 'updated_at': at, 'document': doc})
        references = []
        if approved_only:
            for rec in records:
                sample = deepcopy(self.case(rec['task_id'])['sample'])
                doc = rec['document']
                sample.update(doc['candidate'])
                sample.update(reference_status='human_reviewed_not_expert_gold', expert_signed=False,
                              human_reviewer=doc['reviewer'], human_review_note=doc['case_note'],
                              human_point_reviews=doc['point_reviews'], review_revision=rec['revision'],
                              ner_supervised=True, source_bundle_id=self.index['bundle_id'])
                references.append(sample)
        return {'format': 'emr_human_review_v1', 'bundle_id': self.index['bundle_id'],
                'expert_signed': False, 'approved_only': approved_only,
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'total_cases': len(self.case_ids), 'exported_cases': len(records),
                'records': records, 'references': references}

    def restore(self, package):
        if package.get('format') != 'emr_human_review_v1' or package.get('bundle_id') != self.index['bundle_id']:
            raise ValueError('导入文件的来源批次或格式不匹配')
        records = package.get('records')
        if not isinstance(records, list) or len({r['task_id'] for r in records}) != len(records):
            raise ValueError('导入病例重复或格式不合法')
        with self.lock:
            for rec in records:
                case = self.case(rec['task_id'])
                if rec.get('text_sha256') != case['sample']['text_sha256']:
                    raise ValueError('导入病例原文哈希不匹配')
                validate_document(rec['document'], case, self.index['schema'])
                current = self.get(rec['task_id'])
                if current['revision'] and current['document'] != rec['document']:
                    raise RevisionConflict('导入文件与现有草稿冲突，未覆盖任何草稿')
            # All validation/conflict checks precede mutation; one transaction.
            now = datetime.now(timezone.utc).isoformat()
            imported = 0
            with database(self.db) as con:
                con.execute('BEGIN IMMEDIATE')
                for rec in records:
                    if con.execute('SELECT 1 FROM drafts WHERE task_id=?', (rec['task_id'],)).fetchone():
                        continue
                    encoded = json.dumps(rec['document'], ensure_ascii=False)
                    con.execute('INSERT INTO drafts VALUES(?,?,?,?)', (rec['task_id'], 1, encoded, now))
                    con.execute('INSERT INTO events(task_id,revision,document,created_at) VALUES(?,?,?,?)',
                                (rec['task_id'], 1, encoded, now))
                    imported += 1
            return {'imported': imported, 'unchanged': len(records)-imported}


def serve(root, port):
    store = ReviewStore(root)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, obj, status=200, filename=None):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            if filename:
                self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in ('/', '/review.html', '/review_examples.html'):
                    body = (store.root/'review.html').read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == '/api/index':
                    self.send_json({**store.index, 'review_state': store.summary()})
                elif parsed.path == '/api/case':
                    self.send_json(store.get(parse_qs(parsed.query).get('task_id', [''])[0]))
                elif parsed.path == '/api/history':
                    self.send_json(store.history(parse_qs(parsed.query).get('task_id', [''])[0]))
                elif parsed.path == '/api/export':
                    approved = parse_qs(parsed.query).get('approved', ['0'])[0] == '1'
                    self.send_json(store.export(approved), filename='reviewed_reference.json' if approved else 'review_backup.json')
                else:
                    self.send_json({'error': 'Not found'}, 404)
            except (ValueError, KeyError) as exc:
                self.send_json({'error': str(exc)}, 400)

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    raise ValueError('跨站保存请求被拒绝')
                if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                    raise ValueError('跨站保存请求被拒绝')
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('Expected JSON')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20_000_000:
                    raise ValueError('Request size invalid')
                payload = json.loads(self.rfile.read(length))
                if self.path == '/api/save':
                    self.send_json(store.save(payload['task_id'], payload['revision'], payload['document']))
                elif self.path == '/api/approve':
                    self.send_json(store.approve(payload['task_id'], payload['revision']))
                elif self.path == '/api/import':
                    self.send_json(store.restore(payload))
                else:
                    self.send_json({'error': 'Not found'}, 404)
            except RevisionConflict as exc:
                self.send_json({'error': str(exc)}, 409)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)

        def log_message(self, format, *args):
            # Avoid printing medical content or every autosave to terminal.
            pass

    print(f'REVIEW_TOOL http://127.0.0.1:{port}/review.html ({len(store.case_ids)} cases)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--source-dir', type=Path, required=True)
    build_parser.add_argument('--schema', type=Path, required=True)
    build_parser.add_argument('--output-dir', type=Path, required=True)
    build_parser.add_argument('--sources', type=Path)
    build_parser.add_argument('--review-findings', type=Path)
    server_parser = sub.add_parser('serve')
    server_parser.add_argument('--root', type=Path, required=True)
    server_parser.add_argument('--port', type=int, default=4181)
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.source_dir, args.schema, args.output_dir, args.sources, args.review_findings)
        print(json.dumps(result['counts'], ensure_ascii=False))
    else:
        serve(args.root, args.port)


if __name__ == '__main__':
    main()
