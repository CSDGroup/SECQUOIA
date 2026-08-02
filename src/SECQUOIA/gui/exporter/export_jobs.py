"""The file writing exporters: GIF/animation, TIFF stack, video, PNG."""

import datetime as _dt
import logging
import os

import imageio.v2 as imageio
import numpy as np
import tifffile
from PIL import Image

from SECQUOIA.config import EXPORT
from SECQUOIA.utils.paths import project_analysis_dir

LOG = logging.getLogger(__name__)


class _ExportJobs:
    """The file writing exporters: GIF/animation, TIFF stack, video, PNG."""

    def _export_animation_multi(self, main_window):
        """Export the canvas as an animated GIF/MP4/AVI sequence."""
        panels = getattr(main_window, "panels", [])
        if not panels:
            self._warn(
                main_window, "Export Animation", "No previews to export."
            )
            return
        t0 = int(main_window.start_t_input.value())
        t1 = int(main_window.end_t_input.value())
        if t0 > t1:
            self._warn(
                main_window,
                "Export Animation",
                "Start (t) cannot be greater than End (t).",
            )
            return

        experiment_root = getattr(
            main_window, "experiment_root", None
        ) or getattr(main_window, "folder", os.getcwd())
        save_folder = project_analysis_dir(main_window, experiment_root)
        outdir_root = os.path.join(save_folder, "SECQUOIA_gif")
        os.makedirs(outdir_root, exist_ok=True)

        fps = max(1, int(main_window.gif_speed_input.value()))
        duration_ms = max(10, int(round(1000.0 / float(fps))))
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        chosen = (
            main_window.anim_fmt_combo.currentText().upper()
            if hasattr(main_window, "anim_fmt_combo")
            else "GIF"
        )

        outdir = os.path.join(
            outdir_root, self._active_ident_token(main_window)
        )
        os.makedirs(outdir, exist_ok=True)
        base = self._active_ident_token(main_window)

        frames_rgb, durations_ms = [], []
        for t in range(t0, t1 + 1):
            fr = self._render_canvas_frame(main_window, t)
            if fr is None:
                self._warn(
                    main_window,
                    "Export Animation",
                    "Missing data under 'Strict' policy. Aborting export.",
                )
                return
            frames_rgb.append(fr.convert("RGB"))
            durations_ms.append(duration_ms)

        if not frames_rgb:
            self._warn(
                main_window,
                "Export Animation",
                "No frames to export (all missing per policy).",
            )
            return

        if chosen == "GIF":
            gif_duration_ms, fps_clamped = self._gif_safe_duration_ms(fps)
            frames_p = self._quantize_frames_per_frame(frames_rgb, colors=256)
            gif_path = os.path.join(outdir, f"{base}_{timestamp}.gif")
            frames_p[0].save(
                gif_path,
                save_all=True,
                append_images=frames_p[1:],
                duration=gif_duration_ms,
                loop=0,
                disposal=2,
                optimize=True,
            )
            if fps_clamped:
                self._info(
                    main_window,
                    "Export Animation",
                    f"GIF playback speed is capped at {EXPORT.GIF_SAFE_MAX_FPS} fps by "
                    "the format itself (most viewers force a ~10 fps floor below "
                    f"~20 ms/frame), so this file was saved at "
                    f"{EXPORT.GIF_SAFE_MAX_FPS} fps instead of the requested {fps}. "
                    "Export as MP4 or AVI for faster playback.",
                )
        else:
            if chosen in ("MP4", "AVI"):
                step_ms = int(round(1000.0 / float(fps)))
                expanded = []
                for fr, dur in zip(frames_rgb, durations_ms, strict=False):
                    reps = max(1, int(round(dur / step_ms)))
                    for _ in range(reps):
                        expanded.append(fr)
                frames_rgb = expanded
            self._export_video_files(
                frames_rgb,
                fps,
                outdir,
                f"{base}_{timestamp}",
                want_mp4=(chosen == "MP4"),
                want_avi=(chosen == "AVI"),
            )

        self._show_done_with_open(
            main_window, "Export Animation", f"Saved to:\n{outdir}", outdir
        )

    def _export_tiff_stack(self, main_window):
        """Export the designed canvas layout as a multi-page TIFF stack."""
        panels = getattr(main_window, "panels", [])
        if not panels:
            self._warn(
                main_window,
                "Export TIFF Stack",
                "No previews to export.",
            )
            return

        t0 = int(main_window.start_t_input.value())
        t1 = int(main_window.end_t_input.value())

        if t0 > t1:
            self._warn(
                main_window,
                "Export TIFF Stack",
                "Start (t) cannot be greater than End (t).",
            )
            return

        experiment_root = getattr(
            main_window, "experiment_root", None
        ) or getattr(main_window, "folder", os.getcwd())

        save_folder = project_analysis_dir(main_window, experiment_root)

        outdir_root = os.path.join(save_folder, "SECQUOIA_tiff_stacks")
        outdir = os.path.join(
            outdir_root, self._active_ident_token(main_window)
        )
        os.makedirs(outdir, exist_ok=True)

        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self._active_ident_token(main_window)
        out_path = os.path.join(outdir, f"{base}_{timestamp}_layout_stack.tif")

        frames = []

        for t in range(t0, t1 + 1):
            try:
                frame = self._render_canvas_frame(main_window, t)
                if frame is None:
                    self._warn(
                        main_window,
                        "Export TIFF Stack",
                        f"Missing data at t={t} under Strict policy. "
                        "Use Hold last or Nearest, or adjust the time range.",
                    )
                    return

                arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
                frames.append(arr)

            except (
                KeyError,
                IndexError,
                ValueError,
                RuntimeError,
                OSError,
            ) as e:
                self._stop_play(main_window)
                self._warn(
                    main_window,
                    "Export TIFF Stack",
                    f"Stopped at t={t} while rendering TIFF stack:\n{e}",
                )
                return

        if not frames:
            self._warn(
                main_window,
                "Export TIFF Stack",
                "No frames were rendered.",
            )
            return

        stack = np.stack(frames, axis=0)

        try:
            tifffile.imwrite(
                out_path,
                stack,
                imagej=True,
                photometric="rgb",
                metadata={
                    "axes": "TYXS",
                    "fps": int(main_window.gif_speed_input.value()),
                    "start_t": t0,
                    "end_t": t1,
                },
            )
        except (OSError, ValueError, RuntimeError) as e:
            self._error(
                main_window,
                "Export TIFF Stack",
                f"Failed to save TIFF stack:\n{e}",
            )
            return

        self._show_done_with_open(
            main_window,
            "Export TIFF Stack",
            f"Saved TIFF stack:\n{out_path}",
            outdir,
        )

    def _export_video_files(
        self, frames_rgb, fps, outdir, base_name, want_mp4=True, want_avi=False
    ):
        """Write MP4/AVI files using imageio with proper padding."""

        def _pad_to_mbs(img, m=16):
            w, h = img.size
            W = ((w + m - 1) // m) * m
            H = ((h + m - 1) // m) * m
            if (w, h) == (W, H):
                return img
            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            canvas.paste(img, (0, 0))
            return canvas

        frames_rgb = [_pad_to_mbs(fr.convert("RGB")) for fr in frames_rgb]
        frames_np = [np.asarray(fr, dtype=np.uint8) for fr in frames_rgb]

        def _with_imageio(path, codec):
            """Write all prepared frames to a video file."""
            writer = imageio.get_writer(path, codec=codec, fps=fps)
            for f in frames_np:
                writer.append_data(f)
            writer.close()

        errs, saved_any = [], False
        if want_mp4:
            try:
                _with_imageio(
                    os.path.join(outdir, f"{base_name}.mp4"), "libx264"
                )
                saved_any = True
            except (OSError, RuntimeError, ValueError) as e:
                errs.append(f"MP4: {e}")
        if want_avi:
            try:
                _with_imageio(
                    os.path.join(outdir, f"{base_name}.avi"), "mpeg4"
                )
                saved_any = True
            except (OSError, RuntimeError, ValueError) as e:
                errs.append(f"AVI: {e}")

        if not saved_any:
            msg = "Could not write MP4/AVI with imageio. Please ensure:\n  pip install imageio imageio-ffmpeg"
            if errs:
                msg += "\n\nErrors:\n- " + "\n- ".join(errs)
            raise RuntimeError(msg)

    def _export_single_images(self, main_window):
        """Export single frames per panel or composite canvas images."""
        panels = getattr(main_window, "panels", [])
        if not panels:
            self._warn(
                main_window, "Export Single Images", "No previews to export."
            )
            return
        try:
            t0 = int(main_window.start_t_input.value())
            t1 = int(main_window.end_t_input.value())
        except (TypeError, ValueError) as e:
            self._stop_play(main_window)
            self._warn(
                main_window, "Export Single Images", f"Invalid t-range: {e}"
            )
            return
        if t0 > t1:
            self._warn(
                main_window,
                "Export Single Images",
                "Start (t) cannot be greater than End (t).",
            )
            return

        experiment_root = getattr(
            main_window, "experiment_root", None
        ) or getattr(main_window, "folder", os.getcwd())
        save_folder = project_analysis_dir(main_window, experiment_root)
        outdir = os.path.join(save_folder, "SECQUOIA_single_images")
        os.makedirs(outdir, exist_ok=True)

        fmt = (
            main_window.single_img_fmt_combo.currentText()
            if hasattr(main_window, "single_img_fmt_combo")
            else main_window.export_fmt_combo.currentText()
        ).upper()
        ext = "png" if fmt == "PNG" else "tif"

        if len(panels) == 1:
            p = panels[0]
            ident = (p.get("ident") or "ID").strip()
            ch_txt = (p.get("chan_label_text") or "CH").strip()
            ident_safe = self._safe_name(ident)
            ch_safe = self._safe_name(ch_txt)
            panel_dir = os.path.join(outdir, ident_safe)
            os.makedirs(panel_dir, exist_ok=True)
            exported = 0
            for t in range(t0, t1 + 1):
                try:
                    img = self._render_panel_pil(main_window, p, t).convert(
                        "RGB"
                    )
                except (
                    KeyError,
                    IndexError,
                    ValueError,
                    RuntimeError,
                    OSError,
                ) as e:
                    self._stop_play(main_window)
                    self._warn(
                        main_window,
                        "Export Single Images",
                        f"Stopped at t={t} for ID '{ident}':\n{e}",
                    )
                    return
                fpath = os.path.join(
                    panel_dir, f"{ident_safe}_t{t:05d}_{ch_safe}.{ext}"
                )
                if fmt == "TIFF":
                    img.save(fpath, format="TIFF", compression="tiff_deflate")
                else:
                    img.save(fpath, format="PNG")
                exported += 1
            self._show_done_with_open(
                main_window,
                "Export Single Images",
                f"Exported {exported} images to:\n{panel_dir}",
                panel_dir,
            )
            return

        comp_dir = os.path.join(
            outdir, f"{self._active_ident_token(main_window)}_canvas"
        )
        os.makedirs(comp_dir, exist_ok=True)
        exported = 0
        for ti in range(t0, t1 + 1):
            try:
                canvas_img = self._render_canvas_frame(main_window, ti)
                if canvas_img is None:
                    self._warn(
                        main_window,
                        "Export Single Images",
                        f"Missing data at t={ti} under policy '{getattr(main_window.missing_policy_combo,'currentText',lambda: 'Strict')()}'. Adjust policy or time range.",
                    )
                    return
                canvas = canvas_img.convert("RGB")
            except (ValueError, RuntimeError, OSError) as e:
                self._stop_play(main_window)
                self._warn(
                    main_window,
                    "Export Single Images",
                    f"Stopped at t={ti} while rendering canvas:\n{e}",
                )
                return
            fpath = os.path.join(
                comp_dir,
                f"{self._active_ident_token(main_window)}_canvas_t{ti:05d}.{ext}",
            )
            if fmt == "TIFF":
                canvas.save(fpath, format="TIFF", compression="tiff_deflate")
            else:
                canvas.save(fpath, format="PNG")
            exported += 1
        self._show_done_with_open(
            main_window,
            "Export Single Images",
            f"Exported {exported} canvas images to:\n{comp_dir}",
            comp_dir,
        )
