# -*- coding: utf-8 -*-
"""autogrid.caff —— CAFF（Cubism Archive File Format）解析，内置于工具内。

格式：大端序；文件表与 blob 用 32 位 key 做 XOR 混淆；单个条目可能是
Java ZipOutputStream 风格的 local file header + raw deflate。
规范出处：github.com/umamoorg/umamo docs/formats/CMO3.md
"""
from __future__ import annotations

import struct
import zlib

ZIP_METHOD = {16: "RAW", 33: "FAST(L1)", 37: "SMALL(L5)"}


class _R:
    def __init__(self, b, pos=0):
        self.b, self.p = b, pos

    def skip(self, n):
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def i32(self):
        v = struct.unpack_from(">i", self.b, self.p)[0]
        self.p += 4
        return v


class _X:
    """XOR 解密读取器（大端，无符号掩码）。"""

    def __init__(self, b, pos, key):
        self.b, self.p = b, pos
        self.k = key & 0xFFFFFFFF

    def key8(self):
        return self.k & 0xFF

    def skip(self, n):
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u8(self):
        v = self.b[self.p] ^ self.key8()
        self.p += 1
        return v

    def bool(self):
        return self.u8() != 0

    def u32(self):
        v = struct.unpack_from(">I", self.b, self.p)[0] ^ self.k
        self.p += 4
        return v & 0xFFFFFFFF

    def u64(self):
        raw = struct.unpack_from(">Q", self.b, self.p)[0]
        self.p += 8
        return (raw ^ ((self.k << 32) | self.k)) & 0xFFFFFFFFFFFFFFFF

    def varint(self):
        v = self.u8()
        if v & 0x80:
            raise ValueError("多字节 varint（本实现不支持）")
        return v & 0x7F

    def bytes_(self):
        n = self.varint()
        out = bytes(x ^ self.key8() for x in self.b[self.p:self.p + n])
        self.p += n
        return out

    def string(self):
        return self.bytes_().decode("utf-8", "replace")


def parse(path):
    """解析 .cmo3 容器。返回 (data, key, entries)；entries 每项 dict(i,path,tag,off,size,ob,co)。"""
    data = open(path, "rb").read()
    r = _R(data)
    magic = r.skip(4)
    if magic != b"CAFF":
        raise ValueError("不是 CAFF 容器：magic=%r" % (magic,))
    r.skip(3 + 4 + 3)
    key = r.i32() & 0xFFFFFFFF
    r.skip(8)
    r.skip(1 + 1 + 2)                  # preview: imageFormat / colorType / pad
    r.skip(2 + 2 + 8 + 4 + 8)          # preview: w / h / startPos / fileSize / pad
    x = _X(data, r.p, key)
    count = x.u32()
    entries = []
    for i in range(count):
        p_ = x.string()
        tag = x.string()
        sp_raw = x.u64()
        sz = x.u32()
        ob = x.bool()
        co = x.u8()
        x.skip(8)
        entries.append(dict(i=i, path=p_, tag=tag, off=sp_raw & 0xFFFFFFFF,
                            size=sz, ob=ob, co=co))
    return data, key, entries


def read_entry(data, key, e):
    """取出单条目原始字节（自动 XOR 去混淆 + 解压）。"""
    blob = data[e["off"]:e["off"] + e["size"]]
    if e["ob"]:
        blob = blob.translate(bytes(c ^ (key & 0xFF) for c in range(256)))
    if e["co"] != 16:
        nlen = struct.unpack_from("<H", blob, 26)[0]
        elen = struct.unpack_from("<H", blob, 28)[0]
        blob = zlib.decompressobj(-15).decompress(blob[30 + nlen + elen:])
    return blob
