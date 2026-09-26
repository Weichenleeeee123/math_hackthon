"""不落盘地读远程 zip：用 HTTP Range 按需拉取，zipfile 当成本地文件用。

带宽和磁盘都紧时有用：DOTAv1.zip 有 2GB，评测只要其中约 0.35GB 的 val；
VisDrone 训练集 1.55GB，可以边下边转换，不必先存整个 zip。

实测这条链路每个请求有约 6.6 s 的固定延迟（其中一半是 github.com 的重定向），单次传输约 2 MB/s，
所以：重定向后的签名地址只解析一次（过期后自动重新解析）；按 32MB 对齐分块；
顺序读时后台并行预取后面几块。服务器必须支持 Range（GitHub release 支持）。
"""

import io
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 32 << 20, prefetch: int = 4, retries: int = 6):
        self.url, self.block, self.prefetch, self.retries = url, block, prefetch, retries
        self.pos, self.fetched = 0, 0
        self._lock = threading.Lock()
        self._blocks = {}  # 块号 -> Future[bytes]
        self._pool = ThreadPoolExecutor(max_workers=max(prefetch, 1))
        self._resolve()

    def _resolve(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
            self._direct = r.geturl()  # 重定向之后的签名地址

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _download(self, i):
        start = i * self.block
        end = min(self.size, start + self.block) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self._direct, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) == end - start + 1:
                    with self._lock:
                        self.fetched += len(data)
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404, 410):  # 签名地址过期：重新解析
                    with self._lock:
                        self._resolve()
            except OSError:
                pass
            time.sleep(3 * (attempt + 1))
        raise OSError(f"block {i} ({start}-{end}) of {self.url} failed after {self.retries} tries")

    def _get(self, i):
        n_blocks = (self.size + self.block - 1) // self.block
        with self._lock:
            for j in range(i, min(i + 1 + self.prefetch, n_blocks)):
                if j not in self._blocks:
                    self._blocks[j] = self._pool.submit(self._download, j)
            for j in [j for j in self._blocks if j < i - 1 or j > i + 1 + self.prefetch]:
                del self._blocks[j]  # 只保留当前附近的块，控制内存
            fut = self._blocks[i]
        return fut.result()

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.size:
            i, off = divmod(self.pos, self.block)
            data = self._get(i)
            chunk = data[off: off + n - done]
            b[done: done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()
