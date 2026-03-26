"""Quick smoke test for XY block-tiling helpers."""
from deconvolve import _compute_tile_grid, _compute_tile_slices, _blend_tile
import numpy as np

# Test tile grid computation
print("Grid for 256x256, n=4:", _compute_tile_grid((256, 256), 4))
print("Grid for 256x512, n=4:", _compute_tile_grid((256, 512), 4))
print("Grid for 256x256, n=1:", _compute_tile_grid((256, 256), 1))
print("Grid for 1024x2048, n=8:", _compute_tile_grid((1024, 2048), 8))

# Test tile slices
tiles = _compute_tile_slices((10, 100, 100), 2, 2, overlap=16)
print(f"\nTile count: {len(tiles)}")
for i, t in enumerate(tiles):
    ey, ex = t["extract"][1], t["extract"][2]
    by, bx = t["blend_y"], t["blend_x"]
    print(f"  Tile {i}: Y=[{ey.start}:{ey.stop}] X=[{ex.start}:{ex.stop}]  blend_y={by}  blend_x={bx}")

# Test full round-trip: blend should reconstruct if all tiles are identity
img = np.random.rand(5, 64, 64).astype(np.float32)
tiles = _compute_tile_slices(img.shape, 2, 2, overlap=7)
num = np.zeros_like(img, dtype=np.float64)
den = np.zeros(img.shape, dtype=np.float64)
for desc in tiles:
    tile = img[desc["extract"]].copy()
    w, wm = _blend_tile(tile, desc)
    ext = desc["extract"]
    num[ext] += w.astype(np.float64)
    den[ext] += wm[np.newaxis, :, :].astype(np.float64)
den = np.maximum(den, 1e-12)
result = (num / den).astype(np.float32)
err = np.max(np.abs(result - img))
print(f"\nRound-trip max error (should be ~0): {err:.2e}")
print("PASS" if err < 1e-5 else "FAIL")
