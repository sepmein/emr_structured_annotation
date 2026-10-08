"""Check local links and HTML anchors after moving annotation documents."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.ids = [], set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for attr in ('href', 'src'):
            if attr in attrs:
                self.links.append(attrs[attr])


folder = Path(__file__).resolve().parents[1]
errors = []
for file in sorted(folder.rglob('*')):
    if file.suffix not in ('.html', '.md'):
        continue
    text = file.read_text()
    links = Links(text).links if file.suffix == '.html' else re.findall(r'\]\(([^\s)]+)\)', text)
    for href in links:
        url = urlsplit(href)
        if url.scheme or url.netloc:
            continue
        target = (file.parent / unquote(url.path)).resolve() if url.path else file
        if not target.exists():
            errors.append(f'{file.relative_to(folder)}: missing {href}')
        # Dictionary section anchors are created at runtime and checked in the browser.
        elif url.fragment and target.suffix == '.html' and target.name != 'label-dictionary.html' and unquote(url.fragment) not in Links(target.read_text()).ids:
            errors.append(f'{file.relative_to(folder)}: missing anchor {href}')
assert not errors, '\n'.join(errors)
print('PASS: annotation document files and HTML anchors exist')
