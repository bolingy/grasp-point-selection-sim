import zipfile
import os
import shutil
import trimesh
import random
import glob
import subprocess
import click
import io
import pickle
import time
import itertools
import tempfile

import numpy as np
from datetime import datetime
import random
import string

object_database_path = '/tmp/Google_Scanned_Objects/'

def unzip_file(zip_filepath, dest_folder):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)


def resize_mesh(obj_file_path, scaling_factor):
    # Load the mesh from the file
    mesh = trimesh.load(obj_file_path)
    # Scale the mesh vertices
    mesh.vertices *= scaling_factor
    # Save the resized mesh to a new file (or overwrite the existing one)
    mesh.export('resized_mesh.obj')


def scale_mesh_within_bounds(obj_file_path, output_path, length_bounds=0.08, width_bounds=0.08, height_bounds=0.13):
    # Load the mesh from the file
    mesh = trimesh.load(obj_file_path)
    # Calculate the current dimensions
    length, width, height = mesh.extents
    # Calculate scaling factors for each dimension
    length_scale = length_bounds / length
    width_scale = width_bounds / width
    height_scale = height_bounds / height
    # Take the smallest scaling factor to ensure the mesh fits the bounds without distortion
    scaling_factor = min(1, min(length_scale, width_scale, height_scale))
    # Scale the mesh vertices
    mesh.vertices *= scaling_factor
    # Save the resized mesh to the specified output path
    mesh.export(output_path)
    # Return the new dimensions for verification
    new_length, new_width, new_height = mesh.extents
    return new_length, new_width, new_height


def add_texture_to_mtl(mtl_filepath, temp_dir, texture_filename="texture.png"):
    shutil.move(f"{temp_dir}/materials/textures/texture.png",
                f"{temp_dir}/meshes/texture.png")
    with open(mtl_filepath, 'r') as file:
        content = file.readlines()
    # Append the texture mapping line to the content
    content.append(f"\nmap_Kd {texture_filename}\n")
    # Write the modified content back to the MTL file
    with open(mtl_filepath, 'w') as file:
        file.writelines(content)


def get_mesh_inertia(obj_file_path, mass):
    # Load the mesh from the file
    mesh = trimesh.load(obj_file_path)
    # Calculate the moment of inertia tensor for unit density
    inertia_tensor_unit_density = mesh.moment_inertia
    # Compute the reference mass based on unit density
    m_ref = mesh.volume  # because the density is assumed to be 1
    # Scale the inertia tensor for the given mass
    scaling_factor = mass / m_ref
    inertia_tensor_scaled = scaling_factor * inertia_tensor_unit_density
    # ixx = inertia_tensor_scaled[0, 0]
    # iyy = inertia_tensor_scaled[1, 1]
    # izz = inertia_tensor_scaled[2, 2]
    # ixy = inertia_tensor_scaled[0, 1]
    # ixz = inertia_tensor_scaled[0, 2]
    # iyz = inertia_tensor_scaled[1, 2]
    return inertia_tensor_scaled


def create_urdf_with_inertia(filename, mass, ixx, iyy, izz, ixy=0, ixz=0, iyz=0):
    urdf_content = f"""<robot name="object_model.urdf">
<link name="base_link">
    <visual>
        <origin xyz="0 0 0" rpy="-1.57 0 0"/>
        <geometry>
            <mesh filename="resized_model.obj"/>
        </geometry>
    </visual>
        <material name="white">
        <texture filename="texture.png"/>
    </material>
    <collision>
        <geometry>
            <mesh filename="resized_model.obj"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="{mass}"/>  <!-- You can modify this value as needed -->
        <origin xyz="0 0 0" rpy="0 0 0"/>  <!-- Adjust the origin if needed -->
        <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
    </inertial>
</link>
</robot>
"""
    with open(filename, 'w') as file:
        file.write(urdf_content)


bin_id_resize_bounds = {
    "3F": [0.08, 0.16],
    "3E": [0.075, 0.11],
    "3H": [0.075, 0.12],
}

datetime_string = datetime.now().isoformat().replace(":", "")[:-7]
random_string = ''.join(random.choice(string.ascii_letters)
                        for _ in range(6))


def _get_data_path(bin_id, output_path, runID):
    os.makedirs(f"{output_path}", exist_ok=True)
    temp_path = f"{output_path}/{datetime_string}-{random_string}-grasp_data_v2_{bin_id}/{runID}/"
    return os.path.expanduser(temp_path)


@click.command()
@click.option('--bin-id', type=click.Choice(['3H', '3E', '3F']), default='3F', help='Select bin-id between 3H, 3E and 3F')
@click.option('--num-envs', default=10, help='Enter num-envs as per the gpu capability')
@click.option('--objects-spawn', default=-1, help='Enter objects-spawn for number of objects to be spawned and -1 for all objects to be spawned')
@click.option('--num-runs', default=1, help='Enter num-runs for number of complete runs for each enviornment and for infinite runs enter -1')
def main(bin_id, num_envs, objects_spawn, num_runs):

    if not os.path.isdir(object_database_path):
        print(
            f"ERROR. Unable to find object database at {object_database_path}")
        return

    for runID in range(int(num_runs)) if int(num_runs) != -1 else itertools.count():

        target_base_dir = tempfile.mkdtemp(prefix="google_scanned_models")

        # List all files with the specified extension
        files = glob.glob(os.path.join(object_database_path, '*.zip'))
        if (objects_spawn == -1):
            objects_spawn = len(files)
        # Randomly sample files
        sampled_files = random.sample(files, objects_spawn)

        for name_of_file in sampled_files:
            extract_temp_dir = tempfile.mkdtemp()

            # getting rid of path and '.zip' extension string
            name_of_file = os.path.basename(name_of_file)
            name_of_file = name_of_file[:-4]

            target_object_dir = os.path.join(target_base_dir, name_of_file)
            os.makedirs(target_object_dir)

            unzip_file(
                f"{object_database_path}{name_of_file}.zip", extract_temp_dir)

            bounds = np.random.uniform(*bin_id_resize_bounds[bin_id], size=3)

            scale_mesh_within_bounds(f"{extract_temp_dir}/meshes/model.obj",
                                     f"{extract_temp_dir}/meshes/resized_model.obj", *bounds)

            mass = random.uniform(0.2, 1)

            inertia_tensor_scaled = get_mesh_inertia(
                f"{extract_temp_dir}/meshes/resized_model.obj", mass)
            add_texture_to_mtl(
                f"{extract_temp_dir}/meshes/material.mtl", extract_temp_dir)
            create_urdf_with_inertia(f"{extract_temp_dir}/meshes/model.urdf", mass,
                                     inertia_tensor_scaled[0, 0], inertia_tensor_scaled[1, 1], inertia_tensor_scaled[2, 2])

            for file_name in ["texture.png", "material.mtl", "resized_model.obj", "model.urdf"]:
                shutil.move(f"{extract_temp_dir}/meshes/{file_name}",
                            os.path.join(target_object_dir, file_name))

            shutil.rmtree(extract_temp_dir)

        output_path = f"/tmp/"
        new_dir_path = _get_data_path(bin_id, output_path, runID)
        os.makedirs(new_dir_path, exist_ok=True)

        command = ["python", "data_collection.py", "--bin-id",
                   f"{bin_id}", "--num-envs", f"{num_envs}", "--google-scanned-objects-path", target_base_dir, "--output-path", f"{new_dir_path}"]
        result = subprocess.run(command)

        if result.returncode == 0:
            print("Simulation exited successfully!")
        else:
            print(
                f"Simulation exit failed with return code {result.returncode}.")

        shutil.rmtree(target_base_dir)


if __name__ == "__main__":
    main()
