import argparse
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import test_net
from function import adaptive_instance_normalization, coral
import numpy as np

def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

parser = argparse.ArgumentParser()
# ... (all parser arguments remain the same) ...
parser.add_argument('--content', type=str, help='File path to the content image')
parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--SRSFM', type=str, default='models/SRSFM.pth')
parser.add_argument('--content_size', type=int, default=512, help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512, help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true', help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg', help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output', help='Directory to save the output image(s)')
parser.add_argument('--preserve_color', action='store_true', help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0, help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# ... (path handling logic remains the same) ...
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), 'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = test_net.ModulatedDecoder()
vgg = test_net.vgg
SRSFM=test_net.SRSFM()


decoder.eval()
SRSFM.eval()


SRSFM.load_state_dict(torch.load(args.SRSFM, map_location=device))
decoder.load_state_dict(torch.load(args.decoder, map_location=device))


vgg.load_state_dict(torch.load(args.vgg, map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

network = test_net.Net(vgg,decoder,SRSFM)
network.eval()
network.to(device)



# ========== CHANGE 2: Modify style_transfer to handle loss outputs ==========
def style_transfer(network_model, content, style):
    # The network forward pass now returns three values
    result, loss_c, loss_s = network_model(content, style)
    return result, loss_c, loss_s
# =========================================================================

content_tf = test_transform()
style_tf = test_transform()

# ========== CHANGE 3: Initialize accumulators for calculating average loss ==========
total_content_loss = 0.0
total_style_loss = 0.0
num_images = 0
# ====================================================================================

for content_path in content_paths:
    if do_interpolation:
        style = torch.stack([style_tf(Image.open(str(p)).convert('RGB')) for p in style_paths])
        content = content_tf(Image.open(str(content_path)).convert('RGB')) \
            .unsqueeze(0).expand_as(style)
        
        style = style.to(device)
        content = content.to(device)

        with torch.no_grad():
            # Corrected call to use 'network' and capture losses
            output, loss_c, loss_s = style_transfer(network, content, style)

        # ========== CHANGE 4a: Accumulate losses for the batch ==========
        # The returned loss is a tensor for the whole batch. Use its value.
        total_content_loss += loss_c.item() * content.size(0)
        total_style_loss += loss_s.item() * content.size(0)
        num_images += content.size(0)
        # ================================================================

        output = output.cpu()
        # ... (image saving logic remains the same) ...

    else:  
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)).convert('RGB'))
            style = style_tf(Image.open(str(style_path)).convert('RGB'))

            if args.preserve_color:
                style = coral(style, content)

            content = content.to(device).unsqueeze(0)
            style = style.to(device).unsqueeze(0)

            with torch.no_grad():
                # Corrected call to use 'network' and capture losses
                output, loss_c, loss_s = style_transfer(network, content, style)

            # ========== CHANGE 4b: Accumulate losses for single image ==========
            total_content_loss += loss_c.item()
            total_style_loss += loss_s.item()
            num_images += 1
            # =================================================================

            composite = output[0].cpu()
            composite = torch.clamp(composite, 0, 1)
            composite_name = output_dir / f'{content_path.stem}_{style_path.stem}{args.save_ext}'
            save_image(composite, str(composite_name))

# ========== CHANGE 5: Calculate and print the final average losses ==========
if num_images > 0:
    avg_content_loss = total_content_loss / num_images
    avg_style_loss = total_style_loss / num_images
    
    print("\n" + "="*50)
    print(f"Processing finished for {num_images} image(s).")
    print(f"Average Content Loss: {avg_content_loss:.4f}")
    print(f"Average Style Loss:   {avg_style_loss:.4f}")
    print("="*50)
else:
    print("No images were processed.")
# ==========================================================================