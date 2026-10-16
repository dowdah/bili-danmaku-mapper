"""Read-only finished-media library; importing never schedules a task."""
import json
import math
from pathlib import Path
import time

from ..io import read_json
from .files import VIDEO_SUFFIXES
from .state import identifier
from .subtitles import inspect_subtitles

DEFAULT_STATE = dict(position=0.0, volume=1.0, muted=False, playback_rate=1.0,
                     danmaku_enabled=True, font_size=24, opacity=1.0, show_estimated=False, danmaku_offset=0.0, subtitle_id=None, subtitle_enabled=True, subtitle_offset=0.0, playback_audio_stream=0, preferred_engine='web')


def inspect_danmaku(path):
    path = Path(path)
    if path.suffix.lower() != '.json':
        raise ValueError('请选择已对齐的弹幕 JSON')
    try:
        data = read_json(path)
    except (OSError, ValueError):
        raise ValueError('弹幕 JSON 无法读取，请检查文件格式') from None
    items = data.get('data') if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError('弹幕需要包含 data 数组，不支持映射报告或原始 XML')
    for item in items:
        stamp = item.get('time') if isinstance(item, dict) else None
        if (isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp < 0
                or not isinstance(item.get('text'), str) or item.get('type', 'scroll') not in ('scroll','top','bottom')):
            raise ValueError('弹幕条目需要有效的时间、文本和显示类型')
    return {'count': len(items)}


class PlaybackLibrary:
    def __init__(self, store):
        self.store = store
        self.native_owners = {}
        with store.lock:
            store.db.executescript('''
              CREATE TABLE IF NOT EXISTS playback_items(id TEXT PRIMARY KEY, body TEXT NOT NULL, created REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS playback_feedback(id TEXT PRIMARY KEY, item_id TEXT NOT NULL, position REAL NOT NULL, kind TEXT NOT NULL, created REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS playback_states(id TEXT PRIMARY KEY, body TEXT NOT NULL, updated REAL NOT NULL);
            ''')
            store.db.commit()

    def state(self, key):
        with self.store.lock:
            row = self.store.db.execute('SELECT body FROM playback_states WHERE id=?', (key,)).fetchone()
        return {**DEFAULT_STATE, **(json.loads(row[0]) if row else {})}

    def items(self):
        with self.store.lock:
            rows = self.store.db.execute('SELECT body FROM playback_items ORDER BY created DESC').fetchall()
        items = [json.loads(row[0]) for row in rows]
        for job in self.store.jobs():
            if job['status'] != 'complete' or not (job.get('result') or {}).get('files'):
                continue
            files = job['result']['files']
            title=job['source']['title']
            if title in ('','正片'):
                try:title=Path(self.store.file(files['target'])['path']).stem
                except (KeyError,OSError):pass
            items.append({'id':'job:'+job['id'], 'kind':'job', 'job_id':job['id'], 'title':title,
                          'video_id':files['target'], 'danmaku_id':files['danmaku'], 'report_id':files.get('report'),
                          'created':job['created'], 'parent_id':job.get('parent_id')})
        for item in items:
            item['state'] = self.state(item['id'])
            item['available'] = True
            for key in ('video_id','danmaku_id'):
                if item.get(key):
                    try:
                        self.store.file(item[key])
                    except (KeyError, FileNotFoundError):
                        item['available'] = False
            subtitle_id = item['state'].get('subtitle_id')
            if subtitle_id:
                try:
                    subtitle = self.store.file(subtitle_id)
                    item['subtitle'] = {'id':subtitle_id,'name':Path(subtitle['path']).name,**subtitle['info']}
                except (KeyError, FileNotFoundError):
                    item['subtitle_warning'] = '外挂字幕已移动或丢失，请重新选择；影片仍可播放'
            if not item['available']:
                item['error'] = '影片或弹幕文件已移动、丢失或不可访问，请重新选择'
        return sorted(items, key=lambda item:item['created'], reverse=True)

    def item(self, key):
        for item in self.items():
            if item['id'] == key:
                return item
        raise KeyError('Unknown finished item')

    def import_item(self, video_id, danmaku_id=None, item_id=None):
        video = self.store.file(video_id)
        if video['kind'] not in ('playback_video','artifact') or Path(video['path']).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError('请通过播放器的系统对话框选择影片')
        if danmaku_id:
            comments = self.store.file(danmaku_id)
            if comments['kind'] != 'danmaku':
                raise ValueError('请通过系统对话框选择已对齐的弹幕 JSON')
            inspect_danmaku(comments['path'])
        current = self.item(item_id) if item_id else None
        if current and (current['kind'] != 'manual' or current['video_id'] != video_id):
            raise ValueError('只能为当前手动导入的影片更换弹幕')
        key = item_id or 'manual:'+identifier()
        body = {'id':key, 'kind':'manual', 'title':Path(video['path']).name, 'video_id':video_id,
                'danmaku_id':danmaku_id, 'report_id':None, 'created':current['created'] if current else time.time()}
        with self.store.lock:
            self.store.db.execute('INSERT OR REPLACE INTO playback_items VALUES(?,?,?)',
                                  (key, json.dumps(body, allow_nan=False), body['created']))
            self.store.db.commit()
        return self.item(key)

    def feedback(self, key, position, kind):
        self.item(key)
        key_feedback = identifier()
        with self.store.lock:
            self.store.db.execute('INSERT INTO playback_feedback VALUES(?,?,?,?,?)',
                                  (key_feedback, key, position, kind, time.time()))
            self.store.db.commit()
        return {'id':key_feedback, 'position':position, 'kind':kind}

    def save_state(self, key, values, owner=None):
        self.item(key)
        if self.native_owners.get(key) and owner != self.native_owners[key]:
            values = {k:v for k,v in values.items() if k != 'position'}
        previous = self.state(key)
        if values.get('subtitle_id') and values['subtitle_id'] != previous.get('subtitle_id'):
            subtitle = self.store.file(values['subtitle_id'])
            if subtitle['kind'] != 'subtitle':
                raise ValueError('请通过系统对话框选择字幕文件')
            inspect_subtitles(subtitle['path'])
        # A pagehide keepalive can arrive before older queued PATCH requests.
        # Compare browser write stamps under the same lock as the database update.
        revision = values.get('write_revision')
        values = {name: value for name, value in values.items() if name != 'write_revision'}
        with self.store.lock:
            row = self.store.db.execute('SELECT updated FROM playback_states WHERE id=?', (key,)).fetchone()
            if revision is not None and row and revision <= row[0]:
                return self.state(key)
            state = {**self.state(key), **values}
            self.store.db.execute('INSERT OR REPLACE INTO playback_states VALUES(?,?,?)',
                                  (key, json.dumps(state, allow_nan=False), revision if revision is not None else time.time()*1000000))
            self.store.set_setting('playback_last_item', key)
        return state
