import json
import os
import webbrowser
from typing import Optional, Tuple, cast
import io
import re
from pathlib import Path  # Importing Path from pathlib

import aiohttp
import click
import gradio as gr
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from starlette.responses import Response
from PIL import Image, ImageFilter, ImageEnhance  # Import Pillow for image processing

from .._version import get_versions
from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names
from ..sessions.base import BaseSession


@click.command(  # type: ignore
    name="s",
    help="Run the Rembg HTTP server",
)
@click.option(
    "-p",
    "--port",
    default=7000,
    type=int,
    show_default=True,
    help="Port to run the server on",
)
@click.option(
    "-h",
    "--host",
    default="0.0.0.0",
    type=str,
    show_default=True,
    help="Host address",
)
@click.option(
    "-l",
    "--log_level",
    default="info",
    type=str,
    show_default=True,
    help="Logging level",
)
@click.option(
    "-t",
    "--threads",
    default=None,
    type=int,
    show_default=True,
    help="Number of worker threads",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def s_command(port: int, host: str, log_level: str, threads: int, debug: bool) -> None:
    """
    Command-line interface for running the FastAPI web server.

    This function starts the FastAPI web server with the specified port and log level.
    If the number of worker threads is specified, it sets the thread limiter accordingly.
    """
    sessions: dict[str, BaseSession] = {}
    tags_metadata = [
        {
            "name": "Background Removal",
            "description": "Endpoints that perform background removal with different image sources.",
            "externalDocs": {
                "description": "GitHub Source",
                "url": "https://github.com/danielgatis/rembg",
            },
        },
    ]
    app = FastAPI(
        title="Rembg",
        description="Rembg is a tool to remove images background. That is it.",
        version=get_versions()["version"],
        contact={
            "name": "Daniel Gatis",
            "url": "https://github.com/danielgatis",
            "email": "danielgatis@gmail.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
        },
        openapi_tags=tags_metadata,
        docs_url="/api",
    )

    # Enable CORS for all origins
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def parse_color(color: str) -> Tuple[int, int, int, int]:
        """
        Parse a color string into an RGBA tuple.
        Supports rgba(...) strings or hex color formats.
        """
        try:
            if color.startswith("rgba"):
                # Extract rgba values from the string
                match = re.match(
                    r"rgba\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)",
                    color,
                )
                if match:
                    r, g, b, a = map(float, match.groups())
                    return (
                        int(r),
                        int(g),
                        int(b),
                        int(a * 255),
                    )  # Convert alpha to 0-255
                raise ValueError("Invalid rgba format")
            elif color.startswith("#"):
                # Parse hex color
                color = color.lstrip("#")
                if len(color) == 6:
                    r, g, b = (
                        int(color[:2], 16),
                        int(color[2:4], 16),
                        int(color[4:], 16),
                    )
                    return (r, g, b, 255)
                elif len(color) == 8:
                    r, g, b, a = (
                        int(color[:2], 16),
                        int(color[2:4], 16),
                        int(color[4:6], 16),
                        int(color[6:], 16),
                    )
                    return (r, g, b, a)
            raise ValueError("Unsupported color format")
        except ValueError:
            print(
                f"Invalid color format received: {color}. Defaulting to white (#FFFFFF)."
            )
            return (255, 255, 255, 255)

    def gr_app(app: FastAPI):
        def inference(
            input_image: str,
            model: str,
            a: bool,
            af: int,
            ab: int,
            ae: int,
            apply_bg: bool,
            bg_color: Optional[str],
            brightness: float,
            contrast: float,
            saturation: float,
            om: bool,
            ppm: bool,
            antialias: bool,
            aa_level: int,
            cmd_args: Optional[str],
        ) -> str:
            """
            Process the image by applying background color and antialiasing based on user input.

            Args:
                input_image (str): Path to input image file.
                model (str): Model to use.
                a (bool): Alpha matting flag.
                af (int): Alpha matting foreground threshold.
                ab (int): Alpha matting background threshold.
                ae (int): Alpha matting erosion size.
                apply_bg (bool): Background color checkbox.
                bg_color (Optional[str]): Selected background color hex string.
                brightness: float
                contrast: float
                saturation: float
                om (bool): Only mask flag.
                ppm (bool): Post process mask flag.
                antialias (bool): Whether to apply antialiasing to the output.
                aa_level (int): Level of antialiasing to apply.
                cmd_args (Optional[str]): Extra arguments.

            Returns:
                str: Path to output image file.
            """
            if debug:
                print("Starting inference...")
                print(
                    f"Input image: {input_image}, Apply background: {apply_bg}, Background color: {bg_color}"
                )

            # Extract the filename and format
            input_basename = os.path.basename(input_image)
            input_name, input_ext = os.path.splitext(input_basename)
            input_format = input_ext.lstrip(
                "."
            ).upper()  # Format derived from extension

            if not input_format:
                input_format = "PNG"  # Default to PNG if no format is detected

            output_path = (
                f"{input_name}_processed.{input_format.lower()}"  # Output file
            )

            kwargs = {
                "alpha_matting": a,
                "alpha_matting_foreground_threshold": af,
                "alpha_matting_background_threshold": ab,
                "alpha_matting_erode_size": ae,
                "only_mask": om,
                "post_process_mask": ppm,
            }

            if cmd_args:
                try:
                    kwargs.update(json.loads(cmd_args))
                except json.JSONDecodeError:
                    print("Invalid JSON for additional arguments.")

            try:
                # Initialize the session
                kwargs["session"] = new_session(model, **kwargs)

                # Read and process the input image
                with open(input_image, "rb") as i:
                    input_bytes = i.read()
                    output_bytes = remove(
                        input_bytes, **kwargs
                    )  # Process the background removal
            except Exception as e:
                print(f"Error processing image: {e}")
                return ""

            # Apply the background color and save the output
            try:
                with Image.open(io.BytesIO(output_bytes)) as img:
                    input_format = (
                        img.format or "PNG"
                    )  # Default to PNG if format is unavailable

                    if img.mode != "RGBA":
                        img = img.convert("RGBA")

                    if apply_bg and bg_color:
                        rgba_color = parse_color(bg_color)
                        background = Image.new("RGBA", img.size, rgba_color)
                        img = Image.alpha_composite(background, img)

                    # Apply brightness adjustment
                    if brightness != 1.0:
                        enhancer = ImageEnhance.Brightness(img)
                        img = enhancer.enhance(brightness)

                    # Apply contrast adjustment
                    if contrast != 1.0:
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(contrast)

                    # Apply saturation adjustment
                    if saturation != 1.0:
                        enhancer = ImageEnhance.Color(img)
                        img = enhancer.enhance(saturation)

                    if antialias:
                        for _ in range(aa_level):
                            img = img.filter(ImageFilter.SMOOTH)

                    # Save the image with the original filename and format
                    img.save(output_path, format=input_format)

            except Exception as e:
                print(f"Error saving image: {e}")
                return ""

            if debug:
                print(f"Inference completed. Output saved to {output_path}")

            return os.path.abspath(output_path)

        with gr.Blocks() as interface:
            gr.Markdown("# AKADATA LIMITED - ONNX GPU REMBG with Image Adjustments")

            with gr.Row():
                input_image = gr.Image(type="filepath", label="Input Image")
                output_image = gr.Image(type="filepath", label="Output Image")

            with gr.Row():
                apply_btn = gr.Button("Remove image background")

            with gr.Row():
                fg_threshold = gr.Slider(
                    value=247, minimum=0, maximum=255, label="Foreground Threshold"
                )
                bg_threshold = gr.Slider(
                    value=10, minimum=0, maximum=255, label="Background Threshold"
                )
                erosion_size = gr.Slider(
                    value=0, minimum=0, maximum=255, label="Erosion Size"
                )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=sessions_names,
                    value="birefnet-massive",
                    label="Select Model",
                )
                alpha_matting = gr.Checkbox(value=True, label="Enable Alpha Matting")
                only_mask = gr.Checkbox(value=False, label="Only Mask")
                post_process_mask = gr.Checkbox(value=True, label="Post Process Mask")

            with gr.Row():
                apply_bg = gr.Checkbox(value=False, label="Apply Background Color")
                bg_color = gr.ColorPicker(
                    value="#FFFFFF", label="Select Background Color"
                )

            with gr.Row():
                brightness_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.1,
                    label="Brightness",
                )
                contrast_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.1,
                    label="Contrast",
                )
                saturation_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.1,
                    label="Saturation",
                )

            with gr.Row():
                antialias = gr.Checkbox(value=False, label="Apply Antialiasing")
                antialias_level = gr.Slider(
                    value=1, minimum=1, maximum=5, step=1, label="Antialias Level"
                )

            with gr.Row():
                arguments = gr.Textbox(
                    label="Additional Arguments",
                    placeholder="Enter any additional arguments as JSON",
                )

            apply_btn.click(
                fn=inference,
                inputs=[
                    input_image,
                    model_dropdown,
                    alpha_matting,
                    fg_threshold,
                    bg_threshold,
                    erosion_size,
                    apply_bg,
                    bg_color,
                    brightness_slider,
                    contrast_slider,
                    saturation_slider,
                    only_mask,
                    post_process_mask,
                    antialias,
                    antialias_level,
                    arguments,
                ],
                outputs=[output_image],
            )

        interface.launch(server_name=host, server_port=port)
        app.mount("/gradio", gr.routes.MountGradioApp(interface))

    print(f"To access the API documentation, go to http://{host}:{port}/api")
    print(f"To access the UI, go to http://{host}:{port}")

    uvicorn.run(gr_app(app), host=host, port=port, log_level=log_level)
