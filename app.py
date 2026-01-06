"""
Handwriting Synthesis Web Dashboard

A Gradio-based interactive web interface for generating realistic handwriting
from text input with customizable styles, colors, and neatness.

Run with: python app.py
Access at: http://localhost:7860
"""

import os
import tempfile
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gradio as gr
from demo import Hand
import drawing

# Try to import cairosvg for PNG conversion (optional)
try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    print("Note: cairosvg not installed. PNG export will be disabled.")
    print("Install with: pip install cairosvg")

# Configure logging
logging.basicConfig(level=logging.WARNING)

print("Loading handwriting model... (this may take a moment)")
hand = Hand()
print("Model loaded successfully!")

# Valid characters for input
VALID_CHARS = set(drawing.alphabet)


def validate_text(text):
    """Validate input text and return cleaned version with any issues."""
    lines = text.strip().split('\n')
    issues = []
    cleaned_lines = []

    for i, line in enumerate(lines):
        # Check line length
        if len(line) > 75:
            issues.append(f"Line {i+1} exceeds 75 characters (has {len(line)}). It will be truncated.")
            line = line[:75]

        # Check for invalid characters
        invalid = [c for c in line if c not in VALID_CHARS]
        if invalid:
            issues.append(f"Line {i+1} has invalid characters: {set(invalid)}. They will be removed.")
            line = ''.join(c for c in line if c in VALID_CHARS)

        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)

    return cleaned_lines, issues


def generate_handwriting(text, style, bias, color, stroke_width):
    """Generate handwriting from text input."""
    # Handle Gradio 3.x returning string or tuple from dropdown
    if isinstance(style, str):
        style = -1 if style == "Random" else int(style.replace("Style ", ""))
    elif isinstance(style, (list, tuple)):
        style = style[1] if len(style) > 1 else -1

    print(f"DEBUG: text={repr(text)}, style={style}, bias={bias}, color={color}, stroke_width={stroke_width}")

    if not text or not text.strip():
        return None, None, "Please enter some text."

    # Validate and clean text
    lines, issues = validate_text(text)

    if not lines:
        return None, None, "No valid text to generate. Please check your input."

    # Prepare parameters for each line
    num_lines = len(lines)
    biases = [bias] * num_lines

    # For styles: None means random, otherwise use the style number
    if style < 0:
        styles_param = None
    else:
        styles_param = [style] * num_lines

    stroke_colors = [color] * num_lines
    stroke_widths = [stroke_width] * num_lines

    # Generate to temporary file
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
        temp_svg_path = f.name

    try:
        hand.write(
            filename=temp_svg_path,
            lines=lines,
            biases=biases,
            styles=styles_param,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths
        )

        # Read SVG content
        with open(temp_svg_path, 'r') as f:
            svg_content = f.read()

        # Build status message
        status = f"Generated {num_lines} line(s) of handwriting."
        if issues:
            status += "\n\nWarnings:\n" + "\n".join(f"- {issue}" for issue in issues)

        return svg_content, temp_svg_path, status

    except Exception as e:
        return None, None, f"Error generating handwriting: {str(e)}"


def convert_to_png(svg_path):
    """Convert SVG to PNG for download."""
    if not svg_path or not HAS_CAIROSVG:
        return None

    try:
        png_path = svg_path.replace('.svg', '.png')
        cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2.0)
        return png_path
    except Exception as e:
        print(f"PNG conversion failed: {e}")
        return None


def on_generate(text, style, bias, color, width):
    """Handle generate button click."""
    svg_content, svg_path, status = generate_handwriting(text, style, bias, color, width)

    if svg_content:
        # Wrap SVG for display
        html_output = f'<div style="overflow-x: auto; padding: 10px; background: white; border-radius: 8px; border: 1px solid #ddd;">{svg_content}</div>'
        png_path = convert_to_png(svg_path) if HAS_CAIROSVG else None
        return html_output, svg_path, png_path, status
    else:
        return "", None, None, status


# Build the Gradio interface
with gr.Blocks(
    title="Handwriting Synthesis",
    css="""
    .svg-preview svg {
        max-width: 100%;
        height: auto;
    }
    """
) as app:

    gr.Markdown("""
    # Handwriting Synthesis

    Generate realistic handwriting from text using a neural network trained on real handwriting samples.

    **Tips:**
    - Each line can have up to 75 characters
    - Use the style selector to try different handwriting styles (0-12)
    - Adjust the bias slider: lower = more wild/creative, higher = neater
    """)

    with gr.Row():
        # Left column - inputs
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text to Write",
                placeholder="Enter your text here...\nEach line will be written separately.",
                lines=5,
                max_lines=10
            )

            with gr.Row():
                style_input = gr.Dropdown(
                    label="Handwriting Style",
                    choices=["Random"] + [f"Style {i}" for i in range(13)],
                    value="Random",
                    info="Choose a specific style or random"
                )

                bias_input = gr.Slider(
                    label="Neatness (Bias)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.75,
                    step=0.05,
                    info="0 = wild/creative, 1 = neat/controlled"
                )

            with gr.Row():
                color_input = gr.ColorPicker(
                    label="Stroke Color",
                    value="#000000"
                )

                width_input = gr.Slider(
                    label="Stroke Width",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=0.5
                )

            generate_btn = gr.Button("Generate Handwriting", variant="primary")

        # Right column - output
        with gr.Column(scale=1):
            svg_output = gr.HTML(
                label="Generated Handwriting"
            )

            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

            with gr.Row():
                svg_download = gr.File(
                    label="Download SVG"
                )

                if HAS_CAIROSVG:
                    png_download = gr.File(
                        label="Download PNG"
                    )
                else:
                    png_download = gr.File(
                        label="Download PNG (install cairosvg to enable)",
                        visible=False
                    )

    # Connect the generate button
    generate_btn.click(
        fn=on_generate,
        inputs=[text_input, style_input, bias_input, color_input, width_input],
        outputs=[svg_output, svg_download, png_download, status_output]
    )

    # Example inputs
    gr.Examples(
        examples=[
            ["Hello, World!", "Style 0", 0.75, "#000000", 2],
            ["The quick brown fox\njumps over the lazy dog", "Style 5", 0.5, "#1a5f7a", 2],
            ["Dreams are the seeds\nof reality", "Style 9", 0.8, "#8b4513", 1.5],
        ],
        inputs=[text_input, style_input, bias_input, color_input, width_input],
    )

    gr.Markdown("""
    ---
    **Valid Characters:** Letters (a-z, A-Z), numbers (0-9), and: `space ! " # ' ( ) , - . : ; ?`

    *Based on Alex Graves' handwriting synthesis neural network*
    """)


if __name__ == "__main__":
    print("\nStarting Handwriting Synthesis Dashboard...")
    print("Access at: http://localhost:7860\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
