import sys
import logging

sys.stdin.reconfigure(encoding='utf-8')

logging.basicConfig(
    stream = sys.stderr,
    level = logging.INFO,
    format = '%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)
