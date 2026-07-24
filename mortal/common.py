import socket
import struct
import time
from functools import partial

import mlx.core as mx
import msgpack
import numpy as np
from tqdm.auto import tqdm as orig_tqdm

from checkpoint import model_weights
from config import config


tqdm = partial(orig_tqdm, unit="batch", dynamic_ncols=True, ascii=True)
_ARRAY_EXT = 1
_WIRE_MAGIC = b"MORTAL-MLX"
_WIRE_VERSION = 1
_WIRE_HEADER = _WIRE_MAGIC + struct.pack("<H", _WIRE_VERSION)


def filtered_trimmed_lines(lines):
    return filter(lambda line: line, map(lambda line: line.strip(), lines))


def drain():
    remote = (
        config["online"]["remote"]["host"],
        config["online"]["remote"]["port"],
    )
    while True:
        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {"type": "drain"})
            msg = recv_msg(conn)
        if msg["count"] == 0:
            time.sleep(5)
            continue
        return msg["drain_dir"]


def submit_param(mortal, dqn, is_idle=False):
    remote = (
        config["online"]["remote"]["host"],
        config["online"]["remote"]["port"],
    )
    with socket.socket() as conn:
        conn.connect(remote)
        send_msg(
            conn,
            {
                "type": "submit_param",
                "mortal": model_weights(mortal),
                "dqn": model_weights(dqn),
                "is_idle": is_idle,
            },
        )


def _pack_default(value):
    if isinstance(value, mx.array):
        mx.eval(value)
        value = np.array(value)
    if isinstance(value, np.ndarray):
        shape = value.shape
        contiguous = np.ascontiguousarray(value)
        payload = msgpack.packb(
            (contiguous.dtype.str, shape, contiguous.tobytes()),
            use_bin_type=True,
        )
        return msgpack.ExtType(_ARRAY_EXT, payload)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _unpack_ext(code, payload):
    if code != _ARRAY_EXT:
        return msgpack.ExtType(code, payload)
    dtype, shape, data = msgpack.unpackb(payload, raw=False)
    return (
        np.frombuffer(data, dtype=np.dtype(dtype))
        .reshape(tuple(shape))
        .copy()
    )


def pack_msg(msg) -> bytes:
    payload = msgpack.packb(msg, default=_pack_default, use_bin_type=True)
    return _WIRE_HEADER + payload


def unpack_msg(data: bytes):
    if not data.startswith(_WIRE_MAGIC):
        raise WireProtocolError(
            "Online protocol mismatch: expected the versioned MLX "
            "MessagePack protocol. Upgrade the trainer, server, and workers "
            "together."
        )
    if len(data) < len(_WIRE_HEADER):
        raise WireProtocolError("Truncated Mortal online protocol header")
    (version,) = struct.unpack(
        "<H", data[len(_WIRE_MAGIC) : len(_WIRE_HEADER)]
    )
    if version != _WIRE_VERSION:
        raise WireProtocolError(
            f"Unsupported Mortal online protocol version {version}; "
            f"expected {_WIRE_VERSION}"
        )
    return msgpack.unpackb(
        data[len(_WIRE_HEADER) :],
        raw=False,
        ext_hook=_unpack_ext,
        strict_map_key=False,
    )


def send_msg(conn: socket.socket, msg, packed=False):
    tx = msg if packed else pack_msg(msg)
    conn.sendall(struct.pack("<Q", len(tx)))
    conn.sendall(tx)


def recv_msg(conn: socket.socket, map_location=None):
    del map_location
    (size,) = struct.unpack("<Q", recv_binary(conn, 8))
    return unpack_msg(recv_binary(conn, size))


def recv_binary(conn: socket.socket, size):
    assert size > 0
    ret = bytearray(size)
    buf = memoryview(ret)

    while len(buf) > 0:
        n = conn.recv_into(buf)
        if n == 0:
            raise UnexpectedEOF()
        buf = buf[n:]
    return bytes(ret)


class UnexpectedEOF(Exception):
    def __init__(self):
        super().__init__("unexpected EOF")


class WireProtocolError(ValueError):
    pass
