from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ai_edits import (
    ExtendLayout,
    build_fill_workflow,
    build_sharp_upscale_workflow,
    extend_images,
    extend_layout,
    remove_images,
    smooth_fill,
)
from comfy_client import GenerationParams, InvalidParamsError
from web.gallery import params_from_graph


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


@pytest.mark.parametrize(
    ("size", "sides", "amount", "canvas", "box"),
    [
        ((768, 512), {"left", "right"}, 0.25, (1024, 448), (171, 0, 683, 448)),  # height spans the canvas
        ((512, 512), {"right"}, 0.5, (768, 512), (0, 0, 512, 512)),  # small results grow to the 512 minimum side
        ((768, 512), set(("left", "top", "right", "bottom")), 0.25, (1024, 680), (171, 114, 683, 455)),
        ((2000, 1000), {"bottom"}, 0.2, (1024, 608), (0, 0, 1024, 512)),
    ],
)
def test_extend_layout(size, sides, amount, canvas, box) -> None:
    layout = extend_layout(size, sides, amount, max_side=1024)
    assert layout == ExtendLayout(canvas, box)
    assert all(value % 8 == 0 for value in layout.canvas)


@pytest.mark.parametrize(("sides", "amount"), [(set(), 0.25), ({"up"}, 0.25), ({"left"}, 0.6), ({"left"}, 0.01)])
def test_extend_layout_rejects(sides, amount) -> None:
    with pytest.raises(InvalidParamsError):
        extend_layout((512, 512), sides, amount, max_side=1024)


def test_extend_images_keep_the_picture_and_mask_only_new_area(tmp_path) -> None:
    source = tmp_path / "pic.png"
    Image.new("RGB", (400, 400), (10, 200, 30)).save(source)
    layout = ExtendLayout((800, 400), (200, 0, 400, 400))
    canvas, hard, soft = (np.asarray(_open(data).convert("RGB")) for data in extend_images(source, layout))
    assert canvas.shape == (400, 800, 3)
    assert (canvas[:, 200:600] == (10, 200, 30)).all()
    hard_red, soft_red = hard[..., 0], soft[..., 0]
    assert (hard_red[:, :200] == 255).all() and (hard_red[:, 600:] == 255).all()  # new area is repainted
    assert (hard_red[:, 200:216] == 255).all() and (hard_red[:, 216:584] == 0).all()  # plus a thin overlap strip
    assert (soft_red[:, :200] == 255).all() and (soft_red[:, 300:500] == 0).all()  # middle of the picture stays exact
    assert 0 < soft_red[200, 210] < 255  # the seam is blended


def test_smooth_fill_only_changes_the_hole() -> None:
    image = Image.new("RGB", (64, 64), (200, 50, 50))
    ImageDraw.Draw(image).rectangle((0, 32, 63, 63), fill=(20, 20, 200))
    ImageDraw.Draw(image).rectangle((24, 24, 40, 40), fill=(255, 255, 0))  # the "object"
    hole = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(hole).rectangle((22, 22, 42, 42), fill=255)
    before, after = np.asarray(image), np.asarray(smooth_fill(image, hole))
    outside = np.asarray(hole) == 0
    assert (after[outside] == before[outside]).all()
    assert after[23, 32, 0] > 150 and after[41, 32, 2] > 150  # filled from the nearest side's colour
    assert not ((after[24:41, 24:41] == (255, 255, 0)).all(axis=-1)).any()  # the yellow object is gone


def test_remove_images(tmp_path) -> None:
    source, mask = tmp_path / "pic.png", tmp_path / "mask.png"
    Image.new("RGB", (800, 600), (90, 90, 90)).save(source)
    painted = Image.new("RGB", (800, 600), "black")
    ImageDraw.Draw(painted).rectangle((300, 200, 400, 300), fill="white")
    painted.save(mask)
    filled, hard, soft = (_open(data) for data in remove_images(source, mask, (768, 576)))
    assert filled.size == hard.size == soft.size == (768, 576)
    hard_red = np.asarray(hard.convert("RGB"))[..., 0]
    assert hard_red[192 + 50, 288 + 50] == 255 and hard_red[0, 0] == 0
    assert hard_red[192 - 6, 330] == 255  # grown past the brush stroke
    Image.new("RGB", (800, 600), "black").save(mask)
    with pytest.raises(InvalidParamsError, match="Paint over"):
        remove_images(source, mask, (768, 576))


def test_fill_workflow_keeps_unmasked_pixels_and_prompt_is_readable() -> None:
    params = GenerationParams("empty background", "person", 768, 512, 20, 7.0, seed=3, model="inpaint.safetensors")
    graph = build_fill_workflow("inpaint.safetensors", "src.png", "hard.png", "soft.png", params, 0.75, "ComfyUI_remove")
    sampler = graph["8"]["inputs"]
    assert sampler["denoise"] == 0.75 and sampler["latent_image"] == ["5", 2]
    assert graph["5"]["class_type"] == "InpaintModelConditioning" and graph["5"]["inputs"]["mask"] == ["3", 0]
    composite = graph["11"]["inputs"]
    assert composite["destination"] == ["2", 0] and composite["mask"] == ["4", 0]
    assert graph["9"]["inputs"] == {"images": ["11", 0], "filename_prefix": "ComfyUI_remove"}
    # the gallery can still show and search the settings of these results
    read = params_from_graph(graph, size=(768, 512))
    assert read is not None and read["prompt"] == "empty background" and read["model"] == "inpaint.safetensors"


def test_sharp_upscale_workflow() -> None:
    graph = build_sharp_upscale_workflow("RealESRGAN_x4plus.safetensors", "src.png")
    assert graph["2"]["inputs"] == {"model_name": "RealESRGAN_x4plus.safetensors"}
    assert graph["3"]["inputs"] == {"upscale_model": ["2", 0], "image": ["1", 0]}
    assert graph["9"]["inputs"]["filename_prefix"] == "ComfyUI_upscaled_x4"


def test_paste_at_full_size_keeps_the_original_resolution(tmp_path) -> None:
    import asyncio

    from comfy_client import paste_at_full_size

    class FakeClient:
        def __init__(self) -> None:
            self.uploads: dict[str, bytes] = {}

        async def upload_image(self, path, name):
            self.uploads[name] = path.read_bytes()
            return name

        async def upload_bytes(self, data, name):
            self.uploads[name] = data
            return name

    source, mask = tmp_path / "big.png", tmp_path / "mask.png"
    Image.new("RGB", (1024, 448), (10, 20, 30)).save(source)
    painted = Image.new("RGB", (1024, 448), "black")
    ImageDraw.Draw(painted).rectangle((400, 200, 500, 300), fill="white")
    painted.save(mask)
    params = GenerationParams("x", "y", 768, 336, 20, 7.0, seed=1, model="m")
    graph = build_fill_workflow("m", "src.png", "hard.png", "soft.png", params, 0.75, "ComfyUI_remove")
    client = FakeClient()
    assert asyncio.run(paste_at_full_size(client, graph, source, mask, (768, 336))) == (1024, 448)
    composite = graph["11"]["inputs"]
    assert graph[composite["destination"][0]]["inputs"]["image"] == "studio_full_src.png"
    scale = graph[composite["source"][0]]
    assert scale["class_type"] == "ImageScale" and scale["inputs"]["image"] == ["10", 0]
    assert (scale["inputs"]["width"], scale["inputs"]["height"]) == (1024, 448)
    with Image.open(io.BytesIO(client.uploads["studio_full_mask_soft.png"])) as soft:
        assert soft.size == (1024, 448) and soft.convert("L").getpixel((450, 250)) == 255
        assert soft.convert("L").getpixel((10, 10)) == 0
    # already at the work size: nothing to add
    small = build_fill_workflow("m", "src.png", "hard.png", "soft.png", params, 0.75, "ComfyUI_remove")
    Image.new("RGB", (768, 336)).save(source)
    assert asyncio.run(paste_at_full_size(FakeClient(), small, source, mask, (768, 336))) == (768, 336)
    assert "12" not in small


@pytest.mark.parametrize(
    ("work", "size", "expected"),
    [
        # 1024-wide picture made wider: the original stays 1024 px; the canvas grows past the GPU's 1024
        (ExtendLayout((1024, 296), (171, 0, 683, 296)), (1024, 448), ExtendLayout((1535, 448), (256, 0, 1024, 448))),
        # a small picture was scaled up for the GPU: nothing bigger to make
        (ExtendLayout((768, 512), (0, 0, 512, 512)), (400, 400), None),
        # huge pictures are capped at 2048 px
        (ExtendLayout((1024, 512), (0, 0, 683, 512)), (3072, 2304), ExtendLayout((2048, 1024), (0, 0, 1366, 1024))),
    ],
)
def test_full_size_layout(work, size, expected) -> None:
    from ai_edits import full_size_layout

    assert full_size_layout(work, size) == expected


def test_full_size_overlay(tmp_path) -> None:
    from ai_edits import add_full_size_overlay, full_size_images

    source = tmp_path / "pic.png"
    Image.new("RGB", (1024, 448), (200, 10, 10)).save(source)
    work = ExtendLayout((1024, 296), (171, 0, 683, 296))
    full = ExtendLayout((1535, 448), (256, 0, 1024, 448))
    original, opacity = (np.asarray(_open(data).convert("RGB")) for data in full_size_images(source, work, full))
    assert original.shape == (448, 1024, 3) and (original == (200, 10, 10)).all()
    alpha = opacity[..., 0]
    assert alpha[200, 512] == 255  # the middle is the untouched original
    assert alpha[200, 0] < 128 and alpha[200, 1023] < 128  # its left and right edges fade into the new area
    assert alpha[0, 512] == 255 and alpha[447, 512] == 255  # top and bottom touch the canvas edge: no fade
    params = GenerationParams("x", "y", 1024, 296, 20, 7.0, seed=1, model="m")
    graph = build_fill_workflow("m", "src.png", "hard.png", "soft.png", params, 1.0, "ComfyUI_outpaint")
    add_full_size_overlay(graph, full, "orig.png", "alpha.png")
    assert graph["14"]["inputs"]["image"] == ["11", 0] and graph["14"]["inputs"]["width"] == 1535
    assert graph["15"]["inputs"] | {} == {
        "destination": ["14", 0], "source": ["12", 0], "x": 256, "y": 0, "resize_source": False, "mask": ["13", 0],
    }
    assert graph["9"]["inputs"]["images"] == ["15", 0]


# --- Phase 3: background and faces ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["transparent", "white", "black", "blur"])
def test_background_workflow_modes(mode) -> None:
    from ai_edits import build_background_workflow

    graph = build_background_workflow("src.png", (800, 600), mode)
    assert graph["3"]["class_type"] == "RemoveBackground" and graph["9"]["inputs"]["images"] == ["5", 0]
    assert graph["9"]["inputs"]["filename_prefix"] == "ComfyUI_background"
    if mode == "transparent":
        assert graph["5"]["class_type"] == "JoinImageWithAlpha" and graph["5"]["inputs"]["alpha"] == ["4", 0]
        assert graph["4"]["class_type"] == "InvertMask"  # JoinImageWithAlpha inverts: without this the subject vanishes
    else:
        composite = graph["5"]["inputs"]
        assert composite["source"] == ["1", 0] and composite["mask"] == ["3", 0]  # the original subject on top
    if mode == "white":
        assert graph["4"]["inputs"] == {"width": 800, "height": 600, "batch_size": 1, "color": 0xFFFFFF}
    if mode == "blur":
        assert graph["7"]["inputs"]["sigma"] <= 10  # ImageBlur's limit


def test_background_workflow_paints_a_new_background_at_work_size() -> None:
    from ai_edits import build_background_workflow

    params = GenerationParams("a beach", "people", 768, 576, 25, 7.0, seed=2, model="inpaint.safetensors")
    graph = build_background_workflow("src.png", (1600, 1200), "prompt", params)
    assert graph["11"]["inputs"]["width"] == 768 and graph["4"]["inputs"]["width"] == 1600
    assert graph["20"]["inputs"]["mask"] == ["17", 0] and graph["8"]["inputs"]["denoise"] == 1.0
    assert graph["5"]["inputs"]["source"] == ["1", 0]
    read = params_from_graph(graph, size=(1600, 1200))
    assert read is not None and read["prompt"] == "a beach"
    with pytest.raises(InvalidParamsError):
        build_background_workflow("src.png", (800, 600), "sparkles")


def test_face_boxes_and_crops() -> None:
    from ai_edits import face_boxes, face_crops

    mask = Image.new("L", (800, 600), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((100, 100, 160, 180), fill=255)  # face 1
    draw.ellipse((600, 50, 700, 170), fill=255)  # face 2 (bigger)
    draw.ellipse((400, 400, 405, 405), fill=255)  # a speck: ignored
    boxes = face_boxes(mask)
    assert boxes == [(600, 50, 701, 171), (100, 100, 161, 181)]
    big, small = face_crops(boxes, (800, 600))
    x, y, side, _ = big.box
    assert side == round(121 * 2.2) and y == 0 and x + side <= 800  # pushed inside the picture
    assert small.work == 512 and big.work == 512
    assert face_boxes(Image.new("L", (100, 100), 0)) == []


def test_face_workflow_chains_one_paste_per_face() -> None:
    from ai_edits import FACE_DENOISE, FaceCrop, build_face_workflow

    params = GenerationParams("an old man, detailed face", "ugly", 512, 512, 25, 7.0, seed=10, model="dream.safetensors")
    crops = [FaceCrop((10, 20, 200, 200), 512), FaceCrop((300, 40, 120, 120), 512)]
    graph = build_face_workflow("src.png", crops, [("w0.png", "p0.png"), ("w1.png", "p1.png")], params)
    first, second = graph["109"]["inputs"], graph["129"]["inputs"]
    assert first["destination"] == ["1", 0] and (first["x"], first["y"]) == (10, 20)
    assert second["destination"] == ["109", 0]
    assert graph["120"]["inputs"]["image"] == ["109", 0]  # face 2 is cut from face 1's result
    assert graph["9"]["inputs"]["images"] == ["129", 0]
    assert graph["105"]["inputs"]["denoise"] == FACE_DENOISE and graph["125"]["inputs"]["seed"] == 11
    assert graph["107"]["inputs"]["width"] == 200 and graph["127"]["inputs"]["width"] == 120  # back to crop size
