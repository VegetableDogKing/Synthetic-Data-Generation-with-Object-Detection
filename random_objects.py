# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
#  SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from omni.isaac.kit import SimulationApp
import os
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False, help="Launch script headless, default is False")
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to record")
parser.add_argument("--distractors", type=str, default="warehouse", 
                    help="Options are 'warehouse' (default), 'additional' or None")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_palletjack_data", 
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()

# This is the config used to launch simulation. 
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless, 
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

simulation_app = SimulationApp(launch_config=CONFIG)

import random
import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Semantics
import omni.replicator.core as rep

from omni.isaac.core.utils.semantics import get_semantics
from omni.isaac.core.utils.semantics import add_update_semantics

# Increase subframes if shadows/ghosting appears of moving objects
# See known issues: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator.html#known-issues
rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)

# Target Objects
baseBoard = "omniverse://localhost/btnBoard_USD/baseBoard.usd"
board = "omniverse://localhost/btnBoard_USD/board.usd"
BSwitchBase = "omniverse://localhost/btnBoard_USD/BSwitch.usd"
GRBtnBase = "omniverse://localhost/btnBoard_USD/GRBtn.usd"
GSwitchBase = "omniverse://localhost/btnBoard_USD/GSwitch.usd"
pipe = "omniverse://localhost/btnBoard_USD/pipe.usd"
handle = "omniverse://localhost/btnBoard_USD/handle.usd"
valve = "omniverse://localhost/btnBoard_USD/valve.usd"
RRockBtnBase = "omniverse://localhost/btnBoard_USD/RRockBtn.usd"
RSRBtnBase = "omniverse://localhost/btnBoard_USD/RSRBtn.usd"
RSwitchBase = "omniverse://localhost/btnBoard_USD/RSwitch.usd"
RYPushBtnBase = "omniverse://localhost/btnBoard_USD/RYPushBtn.usd"
screw = "omniverse://localhost/btnBoard_USD/screw.usd"
yBtnBase = "omniverse://localhost/btnBoard_USD/yBtn.usd"
YSRoundBtnBase = "omniverse://localhost/btnBoard_USD/YSRoundBtn.usd"

# The textures which will be randomized for the wall and floor
TEXTURES = ["/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_tile_square_green.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_concrete_aged_with_lines.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stone_painted_grey.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_various.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_carpet_abstract_pattern.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wood_siding_weathered_green.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_animalfur_pattern_greys.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_artificialgrass_green.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_brick_red_stacked.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_fireplace_wall.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_fabric_square_grid.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_gravel_grey_leaves.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_plastic_blue.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stone_red_hatch.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stucco_red_painted.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stucco_smooth_blue.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg"]

def update_semantics(stage, keep_semantics=[]):
    """ Remove semantics from the stage except for keep_semantic classes"""
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            processed_instances = set()
            for property in prim.GetProperties():
                is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(property.GetPath())
                if is_semantic:
                    instance_name = property.SplitName()[1]
                    if instance_name in processed_instances:
                        # Skip repeated instance, instances are iterated twice due to their two semantic properties (class, data)
                        continue
                    
                    processed_instances.add(instance_name)
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    type_attr = sem.GetSemanticTypeAttr()
                    data_attr = sem.GetSemanticDataAttr()


                    for semantic_class in keep_semantics:
                    # Check for our data classes needed for the model
                        if data_attr.Get() == semantic_class:
                            continue
                        else:
                            # remove semantics of all other prims
                            prim.RemoveProperty(type_attr.GetName())
                            prim.RemoveProperty(data_attr.GetName())
                            prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)
    

# needed for loading textures correctly
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path

def full_textures_list():
    full_tex_list = []
    for texture in TEXTURES:
        full_tex_list.append(prefix_with_isaac_asset_server(texture))
        
    return full_tex_list

def random_point_on_hemisphere(min_radius, max_radius):
    radius = random.uniform(min_radius, max_radius)
    phi = random.uniform(0, math.pi)
    theta = random.uniform(0, 2 * math.pi)
    
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = abs(radius * math.cos(phi))
    
    return x, y, z

def generate_points_on_hemisphere(min_radius, max_radius, num_points):
    points = []
    for _ in range(num_points):
        point = random_point_on_hemisphere(min_radius, max_radius)
        points.append(point)
    return points

# This will handle replicator
def run_orchestrator():

    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def main():
    stage = get_current_stage()
    
    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App uppdate {i}..")
        simulation_app.update()

    textures = full_textures_list()
    position_list = generate_points_on_hemisphere(0.5, 1, CONFIG["num_frames"]*2)

    # We only need labels for the exit objects
    update_semantics(stage=stage, keep_semantics=["BSwitch", "GRBtn", "GSwitch", "RRockBtn", "RSRBtn", "RSwitch", "RYPushBtn", "yBtn", "YSRoundBtn", "handle", "valve"])

    # Create camera with Replicator API for gathering data
    cam = rep.create.camera(clipping_range=(0.1, 1000000))
    
    baseBoard_obj = rep.create.from_usd(baseBoard)
    with baseBoard_obj:
        rep.modify.pose(position=(0.02014, -0.013424, -0.00362),)
        
    board_obj = rep.create.from_usd(board)
    with board_obj:
        rep.modify.pose(position=(0, 0, 0))

    screw_obj = rep.create.from_usd(screw)
    with screw_obj:
        rep.modify.pose(position=(0.18376,0.02589,0.00312))

    pipe_obj = rep.create.from_usd(pipe)
    with pipe_obj:
        rep.modify.pose(position=(-0.17035,0.02147,0.0555))
        
    handle_obj = rep.create.from_usd(handle, semantics=[("class", "handle")])
    with handle_obj:
        rep.modify.pose(position=(-0.17035,0.02147,0.0555))
        
    valve_obj = rep.create.from_usd(valve, semantics=[("class", "valve")])
    with valve_obj:
        rep.modify.pose(position=(-0.17035,0.02147,0.0555))
        
    BSwitchBase_obj = rep.create.from_usd(BSwitchBase, semantics=[("class", "BSwitch")], count=2) 
    GRBtnBase_obj = rep.create.from_usd(GRBtnBase, semantics=[("class", "GRBtn")], count=2)
    GSwitchBase_obj = rep.create.from_usd(GSwitchBase, semantics=[("class", "GSwitch")], count=2)
    RRockBtnBase_obj = rep.create.from_usd(RRockBtnBase, semantics=[("class", "RRockBtn")], count=2)
    RSRBtnBase_obj = rep.create.from_usd(RSRBtnBase, semantics=[("class", "RSRBtn")], count=2)
    RSwitchBase_obj = rep.create.from_usd(RSwitchBase, semantics=[("class", "RSwitch")], count=2)
    RYPushBtnBase_obj = rep.create.from_usd(RYPushBtnBase, semantics=[("class", "RYPushBtn")], count=2)
    yBtnBase_obj = rep.create.from_usd(yBtnBase, semantics=[("class", "yBtn")], count=2)
    YSRoundBtnBase_obj = rep.create.from_usd(YSRoundBtnBase, semantics=[("class", "YSRoundBtn")], count=2)

    # trigger replicator pipeline
    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):
        
        with cam:
            rep.modify.pose(position=rep.distribution.choice(position_list),
                            look_at=rep.distribution.uniform((-0.25, -0.2, 0), (0.25, 0.2, 0)))
        
        # Randomize the pose of all the added exits
        
        with BSwitchBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.00725), (0.19952, 0.13783, 0.00725)),
                            scale=(2, 2, 2))

        
        with GRBtnBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.00916), (0.19952, 0.13783, 0.00916)))

        
        with GSwitchBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.00809), (0.19952, 0.13783, 0.00809)))

        
        with RRockBtnBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.01232), (0.19952, 0.13783, 0.01232)))

        
        with RSRBtnBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.00762), (0.19952, 0.13783, 0.00762)))

        
        with RSwitchBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.01281), (0.19952, 0.13783, 0.01281)))

        
        with RYPushBtnBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.01231), (0.19952, 0.13783, 0.01231)))

        
        with yBtnBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.00933), (0.19952, 0.13783, 0.00933)))

        
        with YSRoundBtnBase_obj:
            rep.modify.pose(position=rep.distribution.uniform((-0.11888, -0.13903, 0.00649), (0.19952, 0.13783, 0.00649)))

        # Randomize the lighting of the scene
        lights = rep.create.light(
            light_type="Dome",
            rotation= (270,0,0),
            texture=rep.distribution.choice(textures)
            )
        
        random_baseBoard = rep.create.material_omnipbr(diffuse_texture=rep.distribution.choice(textures),
                                                    roughness=rep.distribution.uniform(0, 1),
                                                    metallic=rep.distribution.choice([0, 1]),
                                                    emissive_texture=rep.distribution.choice(textures),
                                                    emissive_intensity=rep.distribution.uniform(0, 1000),)
        
        random_Board = rep.create.material_omnipbr(diffuse_texture=rep.distribution.choice(textures),
                                                    roughness=rep.distribution.uniform(0, 1),
                                                    metallic=rep.distribution.choice([0, 1]),
                                                    emissive_texture=rep.distribution.choice(textures),
                                                    emissive_intensity=rep.distribution.uniform(0, 1000),)

        # Randomize the texture of the board
        with rep.get.prims(path_pattern="Cube_024"):
            rep.randomizer.materials(random_baseBoard)
        
        with rep.get.prims(path_pattern="Cube_008"):
            rep.randomizer.materials(random_Board)
        
    # Set up the writer
    writer = rep.WriterRegistry.get("KittiWriter")

    # output directory of writer
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)

    # use writer for bounding boxes, rgb and segmentation
    writer.initialize(output_dir=output_directory,
                    omit_semantic_type=True,)


    # attach camera render products to wrieter so that data is outputted
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product  = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    # run rep pipeline
    run_orchestrator()
    simulation_app.update()



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
