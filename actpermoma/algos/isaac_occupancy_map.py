import omni
from omni.isaac.occupancy_map import _occupancy_map
physx = omni.physx.acquire_physx_interface()
stage_id = omni.usd.get_context().get_stage_id()
stage = omni.usd.get_context().get_stage()
from pxr import UsdPhysics, UsdGeom, Gf
map_generator = _occupancy_map.Generator(physx, stage_id)

omi = _occupancy_map.acquire_occupancy_map_interface()
omi.set_cell_size(0.25)
omi.update()
omi.set_transform((-1, 0, 0), (-8, -8, 0.0), (8, 8, 1.0))
omi.update()
# cell size, output buffer will have 1 for occupied cells, 0 for unoccupied, and 2 for cells that cannot be seen
cell_size = 0.25
map_generator.update_settings(cell_size, 1, 0, 2)
# Set location to map from and the min and max bounds to map to
map_generator.set_transform((-1, 0, 0), (-8, -8, 0.0), (8, 8, 1.0))
mammut_path = '/World/envs/env_0/obstacle/mammut'
cubePath = '/World/occupiedCube'
occupiedCube = UsdGeom.Cube(stage.DefinePrim(cubePath, "Cube"))
occupiedCube.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
occupiedCube.AddTranslateOp().Set(Gf.Vec3f(-5, 0, (1.0 / 2.0)))
occupiedCube.CreateSizeAttr(1.0)
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cubePath))
for i in range(10): self._env._world.step()
import pdb; pdb.set_trace()
omi.generate()
minb = omi.get_min_bound()
maxb = omi.get_max_bound()
omi.get_occupied_positions()

UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(mammut_path))
map_generator.generate2d()
# Get locations of the occupied cells in the stage
points = map_generator.get_occupied_positions()
# Get computed 2d occupancy buffer
buffer = map_generator.get_buffer()
# Get dimensions for 2d buffer
dims = map_generator.get_dimensions()
# import pdb; pdb.set_trace()
# # Debug: See occupancy map
# unknown_col = np.array([0.5, 0.5, 0.5, 1.0]) * 255
# occupied_col = np.array([0.0, 0.0, 0.0, 1.0]) * 255
# freespace_col = np.array([1.0, 1.0, 1.0, 1.0]) * 255
# image = unknown_col * dims[0] * dims[1]
# idx = 0
# for b in buffer:
#     if b == 1.0:
#         image[idx * 4 + 0] = occupied_col[0]
#         image[idx * 4 + 1] = occupied_col[1]
#         image[idx * 4 + 2] = occupied_col[2]
#         image[idx * 4 + 3] = occupied_col[3]
#     if b == 0.0:
#         image[idx * 4 + 0] = freespace_col[0]
#         image[idx * 4 + 1] = freespace_col[1]
#         image[idx * 4 + 2] = freespace_col[2]
#         image[idx * 4 + 3] = freespace_col[3]
#     idx += 1
# from PIL import Image
# im = Image.frombytes("RGBA", (dims.x, dims.y), bytes(image))
# file = "test_occ_image.png"
# folder = "test_occ_image"
# file = file if file[-4:].lower() == ".png" else "{}.png".format(file)
# print("Saving occupancy map image to", folder + "/" + file)
# im.save(folder + "/" + file)
import pdb; pdb.set_trace()