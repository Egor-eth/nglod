# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import moviepy.editor as mpy
from scipy.spatial.transform import Rotation as R
import pyexr

from lib.renderer import Renderer
from lib.models import *
from lib.tracer import *
from lib.options import parse_options
from lib.geoutils import sample_unif_sphere, sample_fib_sphere, normalized_slice
import json


def write_exr(path, data):
    pyexr.write(path, data,
                channel_names={'normal': ['X', 'Y', 'Z'],
                               'x': ['X', 'Y', 'Z'],
                               'view': ['X', 'Y', 'Z']},
                precision=pyexr.HALF)


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--img-dir', type=str, default='_results/render_app/imgs',
                           help='Directory to output the rendered images')
    app_group.add_argument('--render-2d', action='store_true',
                           help='Render in 2D instead of 3D')
    app_group.add_argument('--exr', action='store_true',
                           help='Write to EXR')
    app_group.add_argument('--r360', action='store_true',
                           help='Render a sequence of spinning images.')
    app_group.add_argument('--rsphere', action='store_true',
                           help='Render around a sphere.')
    app_group.add_argument('--nb-poses', type=int, default=64,
                           help='Number of poses to render for sphere rendering.')
    app_group.add_argument('--cam-radius', type=float, default=4.0,
                           help='Camera radius to use for sphere rendering.')
    app_group.add_argument('--disable-aa', action='store_true',
                           help='Disable anti aliasing.')
    app_group.add_argument('--export', type=str, default=None,
                           help='Export model to C++ compatible format.')
    app_group.add_argument('--rotate', type=float, default=None,
                           help='Rotation in degrees.')
    app_group.add_argument('--depth', type=float, default=0.0,
                           help='Depth of 2D slice.')
    app_group.add_argument('--from-file', type=str, default=None,
                           help='Camera settings file.')
    app_group.add_argument('--camera', type=float, nargs=7, default=None,
                           help='Camera settings.')
    app_group.add_argument('--light', type=float, nargs=4, default=[1, -1, 1, 10],
                           help='Point light source origin')
    app_group.add_argument('--bg-color', type=float, nargs=3, default=[1, 1, 1],
                           help='Background color')
    args = parser.parse_args()

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Get model name
    if args.pretrained is not None:
        name = args.pretrained.split('/')[-1].split('.')[0]
    else:
        assert False and "No network weights specified!"

    net = globals()[args.net](args)
    if args.jit:
        net = torch.jit.script(net)

    net.load_state_dict(torch.load(args.pretrained))

    net.to(device)
    net.eval()

    print("Total number of parameters: {}".format(sum(p.numel() for p in net.parameters())))

    param_size = sum(p.nelement() * p.element_size() for p in net.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in net.buffers())

    print(f"Model uncompressed size: {(param_size + buffer_size) / 1024 ** 2 : .3f} MB")

    if args.export is not None:
        net = SOL_NGLOD(net)

        net.save(args.export)
        sys.exit()

    if args.sol:
        net = SOL_NGLOD(net)

    if args.lod is not None:
        net.lod = args.lod

    # Make output directory
    ins_dir = os.path.join(args.img_dir, name)
    if not os.path.exists(ins_dir):
        os.makedirs(ins_dir)

    for t in ['normal', 'rgb', 'exr']:
        _dir = os.path.join(ins_dir, t)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    tracer = globals()[args.tracer](args)
    renderer = Renderer(tracer, args=args, device=device)

    if args.rotate is not None:
        rad = np.radians(args.rotate)
        model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([0, 1, 0])).as_matrix())

        
    else:
        model_matrix = torch.eye(3)

    if args.from_file is not None or args.camera is not None:


        if args.from_file is not None:
            with open(args.from_file) as f:
                camera_file = json.load(f)["camera"]
                


            matrix = torch.tensor(list(map(float, camera_file["matrix"]))).reshape(4, 4)
            
            aspect_ratio = args.render_res[0] / args.render_res[1]

            m_inv = torch.linalg.inv(matrix)

            rot_matrix = m_inv[:3, :3]
            pos = m_inv[3, :3]
            lookat = -m_inv[2, :3]
            up = m_inv[1, :3]

            #rot_matrix.T

            yfov = camera_file["yfov"]
            yfov = np.rad2deg(yfov)


            fromvec = pos
            tovec = pos + lookat #torch.tensor([0.0, 0.0, 0.0])

        else:
            fromvec = torch.FloatTensor(args.camera[:3])
            lookat = torch.FloatTensor(args.camera[3:6])
            tovec = fromvec + lookat
            up = torch.FloatTensor([0, 1, 0])
            yfov = args.camera[6]

        model_matrix = torch.eye(3)

        print(f"From {fromvec} with direction {lookat} with fov {yfov}")

        out = renderer.shade_images(net=net,
                                    f=fromvec,
                                    t=tovec,
                                    fov=yfov,
                                    aa=not args.disable_aa,
                                    mm=model_matrix,
                                    u=up,
                                    bg=args.bg_color,
                                    lp=torch.FloatTensor(args.light))

        data = out.float().numpy().exrdict()

        if args.exr:
            write_exr('{}/exr/{:06d}.exr'.format(ins_dir, p), data)

        img_out = out.image().byte().numpy()
        Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, 0), mode='RGB')
        Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, 0), mode='RGB')

    elif args.r360:
        for angle in np.arange(0, 360, 2):
            rad = np.radians(angle)
            model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([0, 1, 0])).as_matrix())

            out = renderer.shade_images(net=net,
                                        f=args.camera_origin,
                                        t=args.camera_lookat,
                                        fov=args.camera_fov,
                                        aa=not args.disable_aa,
                                        bg=args.bg_color,
                                        mm=model_matrix)


            data = out.float().numpy().exrdict()

            idx = int(math.floor(100 * angle))

            if args.exr:
                write_exr('{}/exr/{:06d}.exr'.format(ins_dir, idx), data)

            img_out = out.image().byte().numpy()
            Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, idx), mode='RGB')
            Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, idx), mode='RGB')

    elif args.rsphere:
        views = sample_fib_sphere(args.nb_poses)
        cam_origins = args.cam_radius * views
        for p, cam_origin in enumerate(cam_origins):
            out = renderer.shade_images(net=net,
                                        f=cam_origin,
                                        t=args.camera_lookat,
                                        fov=args.camera_fov,
                                        aa=not args.disable_aa,
                                        bg=args.bg_color,
                                        mm=model_matrix)

            data = out.float().numpy().exrdict()

            if args.exr:
                write_exr('{}/exr/{:06d}.exr'.format(ins_dir, p), data)

            img_out = out.image().byte().numpy()
            Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, p), mode='RGB')
            Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, p), mode='RGB')

    else:

        out = renderer.shade_images(net=net,
                                    f=args.camera_origin,
                                    t=args.camera_lookat,
                                    fov=args.camera_fov,
                                    aa=not args.disable_aa,
                                    bg=args.bg_color,
                                    mm=model_matrix)


        #print(f"from {args.camera_origin} to {args.camera_lookat}")

        data = out.float().numpy().exrdict()

        if args.render_2d:
            depth = args.depth
            data['sdf_slice'] = renderer.sdf_slice(depth=depth)
            data['rgb_slice'] = renderer.rgb_slice(depth=depth)
            data['normal_slice'] = renderer.normal_slice(depth=depth)

        if args.exr:
            write_exr(f'{ins_dir}/out.exr', data)

        img_out = out.image().byte().numpy()

        Image.fromarray(img_out.rgb).save('{}/{}_rgb.png'.format(ins_dir, name), mode='RGB')
        Image.fromarray(img_out.depth).save('{}/{}_depth.png'.format(ins_dir, name), mode='RGB')
        Image.fromarray(img_out.normal).save('{}/{}_normal.png'.format(ins_dir, name), mode='RGB')
        Image.fromarray(img_out.hit).save('{}/{}_hit.png'.format(ins_dir, name), mode='L')
