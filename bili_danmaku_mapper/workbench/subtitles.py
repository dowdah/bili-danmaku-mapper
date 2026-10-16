"""Read local subtitle text as data; never execute or serve subtitle markup as HTML."""
from html.parser import HTMLParser
from pathlib import Path
import re

MAX_BYTES = 20 * 1024 * 1024
MAX_CUES = 100000
SUFFIXES = {'.srt', '.vtt', '.ass', '.ssa'}


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.suppressed = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.suppressed += 1
        elif tag == 'br' and not self.suppressed:
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.suppressed:
            self.suppressed -= 1
    def handle_data(self, value):
        if not self.suppressed:
            self.parts.append(value)


def plain_text(value):
    parser = PlainText()
    parser.feed(value)
    return ''.join(parser.parts).strip()


def timestamp(value):
    match = re.fullmatch(r'(\d{1,3}):(\d{2})(?::(\d{2}))?([.,])(\d{1,3})', value.strip())
    if not match:
        raise ValueError('字幕时间格式无效')
    first, second, third, _, fraction = match.groups()
    first, second = int(first), int(second)
    if second >= 60 or (third is not None and int(third) >= 60):
        raise ValueError('字幕时间的分秒超出范围')
    seconds = first*3600 + second*60 + int(third) if third is not None else first*60 + second
    return seconds + int(fraction) / (10 ** len(fraction))


def ass_text(value):
    parts, drawing, at = [], False, 0
    for match in re.finditer(r'\{[^}]*\}', value):
        if not drawing:
            parts.append(value[at:match.start()])
        commands = re.findall(r'\\p(\d+)', match[0])
        if commands:
            drawing = int(commands[-1]) != 0
        at = match.end()
    if not drawing:
        parts.append(value[at:])
    return ''.join(parts).replace('\\N', '\n').replace('\\n', '\n').replace('\\h', ' ').strip()


def read_subtitles(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError('请选择 SRT、VTT、ASS 或 SSA 字幕')
    with path.open('rb') as file:
        raw = file.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError('字幕文件超过 20 MiB 限制')
    encodings = ['utf-32'] if raw.startswith((b'\xff\xfe\0\0', b'\0\0\xfe\xff')) else ['utf-16'] if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else ['utf-8-sig', 'gb18030']
    text = None
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeError:
            continue
    if text is None or '\0' in text:
        raise ValueError('无法识别字幕编码，请另存为 UTF-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    cues = []
    def add(start, end, body):
        start, end = timestamp(start), timestamp(end)
        if end <= start:
            raise ValueError('字幕结束时间必须晚于开始时间')
        if body.strip():
            cues.append({'start': start, 'end': end, 'text': body.strip()})
        if len(cues) > MAX_CUES:
            raise ValueError('字幕条目超过 100000 条限制')
    if suffix in ('.ass', '.ssa'):
        section = ''
        fields = ['layer','start','end','style','name','marginl','marginr','marginv','effect','text']
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                section = line.lower()
            if section != '[events]':
                continue
            if line.lower().startswith('format:'):
                fields = [f.strip().lower() for f in line.split(':', 1)[1].split(',')]
                if not all(k in fields for k in ('start','end','text')) or fields[-1] != 'text':
                    raise ValueError('不支持此 ASS/SSA 事件格式，Text 字段需要位于最后')
            elif line.lower().startswith('dialogue:'):
                values = line.split(':', 1)[1].lstrip().split(',', len(fields)-1)
                if len(values) != len(fields):
                    raise ValueError('ASS/SSA 对白格式不完整')
                entry = dict(zip(fields, values))
                add(entry['start'], entry['end'], ass_text(entry['text']))
    else:
        if suffix == '.vtt' and not text.lstrip('\ufeff').startswith('WEBVTT'):
            raise ValueError('VTT 字幕缺少 WEBVTT 文件头')
        for block in re.split(r'\n[ \t]*\n', text.strip()):
            lines = block.splitlines()
            if not lines or lines[0].startswith(('WEBVTT','NOTE','STYLE','REGION')):
                continue
            index = 0 if '-->' in lines[0] else 1
            if index >= len(lines) or '-->' not in lines[index]:
                raise ValueError('字幕条目缺少有效时间轴')
            times = re.fullmatch(r'\s*(\S+)\s*-->\s*(\S+)(?:[ \t]+.*)?', lines[index])
            if not times:
                raise ValueError('字幕时间轴无效')
            add(times[1], times[2], plain_text('\n'.join(lines[index+1:])))
    if not cues:
        raise ValueError('字幕中没有可显示的对白')
    cues.sort(key=lambda cue:(cue['start'], cue['end']))
    note = 'ASS/SSA 按对白文本显示；特效、样式与定位不保留。' if suffix in ('.ass','.ssa') else ''
    return {'cues': cues, 'format': suffix[1:].upper(), 'encoding': encoding, 'note': note}


def inspect_subtitles(path):
    result = read_subtitles(path)
    return {key: value for key, value in result.items() if key != 'cues'} | {'count': len(result['cues'])}
