class Hash:
    @staticmethod
    def hash(n: int, s: int, max_val: int) -> int:
        m = (n - s) & 0xFFFFFFFF
        m ^= (m << 2) & 0xFFFFFFFF
        m ^= (m << 3) & 0xFFFFFFFF
        m ^= (m >> 5) & 0xFFFFFFFF
        m ^= (m >> 7) & 0xFFFFFFFF
        m ^= (m << 11) & 0xFFFFFFFF
        m ^= (m << 13) & 0xFFFFFFFF
        m ^= (m >> 17) & 0xFFFFFFFF
        m ^= (m << 19) & 0xFFFFFFFF
        m += s
        m &= 0xFFFFFFFF
        return ((m * max_val) >> 32) & 0xFFFFFFFF

    @staticmethod
    def strings_hash(in_val: int, strs: list[str]) -> int:
        out = in_val
        for s in strs:
            out = Hash.string_hash(out, s)
        return out

    @staticmethod
    def string_hash(in_val: int, s: str) -> int:
        out = in_val
        for c in s:
            out = Hash.hash(out, ord(c), 0xFFFFFFFF)
        return out
