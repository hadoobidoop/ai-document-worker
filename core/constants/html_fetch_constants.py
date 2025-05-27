# document_ingestion/html_fetch_constants.py
UNWANTED_HTML_TAGS = [
    'script', 'style', 'header', 'footer', 'nav', 'aside',
    'form', 'iframe', 'noscript', 'meta', 'link', 'img',
    'audio', 'video', 'canvas', 'svg'
]