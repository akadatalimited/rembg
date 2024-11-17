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
from PIL import Image, ImageFilter  # Import Pillow for image processing

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
        Supports hex color strings (e.g., "#RRGGBB" or "#RRGGBBAA") or rgba strings (e.g., "rgba(r, g, b, a)").
        """
        if color.startswith("rgba"):
            match = re.match(r'rgba\((\d+(\.\d+)?),\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?),\s*([0-1](?:\.\d+)?)\)', color)
            if match:
                r, g, b, a = match.groups()[0], match.groups()[2], match.groups()[4], match.groups()[6]
                return (
                    int(float(r)),  # Convert red to integer
                    int(float(g)),  # Convert green to integer
                    int(float(b)),  # Convert blue to integer
                    int(float(a) * 255)  # Normalize alpha to 0-255
                )
            else:
                print("Invalid rgba color format. Defaulting to #FFFFFF.")
                return (255, 255, 255, 255)
        elif re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$', color):
            color = color.lstrip('#')
            if len(color) == 6:
                r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
                return (r, g, b, 255)
            elif len(color) == 8:
                r, g, b, a = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), int(color[6:8], 16)
                return (r, g, b, a)
        print("Invalid color format. Defaulting to #FFFFFF.")
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
                bg_color (Optional[str]): Selected background color hex string or rgba string.
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
                print(f"Input image: {input_image}")
                print(f"Model: {model}")
                print(f"Alpha matting: {a}")
                print(f"Foreground threshold: {af}")
                print(f"Background threshold: {ab}")
                print(f"Erosion size: {ae}")
                print(f"Apply background: {apply_bg}")
                print(f"Background color: {bg_color}")
                print(f"Only mask: {om}")
                print(f"Post process mask: {ppm}")
                print(f"Antialias: {antialias}")
                print(f"Antialias level: {aa_level}")
                print(f"Command arguments: {cmd_args}")

            output_path = "output.png"
            kwargs = {
                "alpha_matting": a,
                "alpha_matting_foreground_threshold": af,
                "alpha_matting_background_threshold": ab,
                "alpha_matting_erode_size": ae,
                "only_mask": om,
                "post_process_mask": ppm
            }

            if apply_bg and bg_color:
                try:
                    rgba_color = parse_color(bg_color)  # Returns (R, G, B, A)
                except ValueError as e:
                    print(e)
                    rgba_color = (255, 255, 255, 255)  # Default to white with full opacity
                kwargs["bgcolor"] = rgba_color

            if cmd_args:
                try:
                    kwargs.update(json.loads(cmd_args))
                except json.JSONDecodeError:
                    print("Invalid JSON for additional arguments.")

            kwargs["session"] = new_session(model, **kwargs)

            try:
                with open(input_image, "rb") as i:
                    input_bytes = i.read()
                    # Process background removal on input image
                    output_bytes = remove(input_bytes, **kwargs)
            except Exception as e:
                print(f"Error processing image: {e}")
                return ""

            # Load the output image with PIL
            try:
                with Image.open(io.BytesIO(output_bytes)) as img:
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')  # Ensure image is RGBA to handle background

                    if apply_bg and "bgcolor" in kwargs:
                        # Create a background image with the selected color
                        background = Image.new("RGBA", img.size, kwargs["bgcolor"])
                        # Composite the foreground over the background
                        composite = Image.alpha_composite(background, img)
                        # Save the composited image
                        composite.save(output_path, format="PNG")
                    else:
                        if antialias:
                            for _ in range(aa_level):
                                img = img.filter(ImageFilter.SMOOTH)
                        # Save the image as is
                        img.save(output_path, format="PNG")
            except Exception as e:
                print(f"Error saving image: {e}")
                return ""

            if debug:
                print(f"Inference completed. Output saved to {output_path}")

            return os.path.abspath(output_path)

        with gr.Blocks() as interface:
            gr.Markdown("# AKADATA LIMITED - REMCUDABG with Antialiasing")

            with gr.Row():
                input_image = gr.Image(type="filepath", label="Input Image")
                output_image = gr.Image(type="filepath", label="Output Image")

            with gr.Row():
                apply_btn = gr.Button("Remove image background")

            with gr.Row():
                fg_threshold = gr.Slider(value=247, minimum=0, maximum=255, label="Foreground Threshold")
                bg_threshold = gr.Slider(value=10, minimum=0, maximum=255, label="Background Threshold")
                erosion_size = gr.Slider(value=0, minimum=0, maximum=255, label="Erosion Size")

            with gr.Row():
                model_dropdown = gr.Dropdown(choices=sessions_names, value="birefnet-massive", label="Select Model")
                alpha_matting = gr.Checkbox(value=True, label="Enable Alpha Matting")
                only_mask = gr.Checkbox(value=False, label="Only Mask")
                post_process_mask = gr.Checkbox(value=True, label="Post Process Mask")

            with gr.Row():
                apply_bg_checkbox = gr.Checkbox(value=False, label="Apply Background Color")
                bg_color_picker = gr.ColorPicker(value="#FFFFFF", label="Select Background Color")

            with gr.Row():
                antialias_checkbox = gr.Checkbox(value=False, label="Apply Antialiasing")
                aa_level_slider = gr.Slider(value=1, minimum=1, maximum=5, step=1, label="Antialias Level")

            with gr.Row():
                arguments = gr.Textbox(label="Additional Arguments", placeholder="Enter any additional arguments as JSON")

            apply_btn.click(
                fn=inference,
                inputs=[
                    input_image,
                    model_dropdown,
                    alpha_matting,
                    fg_threshold,
                    bg_threshold,
                    erosion_size,
                    apply_bg_checkbox,
                    bg_color_picker,
                    only_mask,
                    post_process_mask,
                    antialias_checkbox,
                    aa_level_slider,
                    arguments
                ],
                outputs=[output_image],
            )

        interface.launch(server_name="0.0.0.0", server_port=port)
        app.mount("/gradio", gr.routes.MountGradioApp(interface))

    print(f"To access the API documentation, go to http://{host}:{port}/api")
    print(f"To access the UI, go to http://{host}:{port}")

    uvicorn.run(gr_app(app), host=host, port=port, log_level=log_level)
