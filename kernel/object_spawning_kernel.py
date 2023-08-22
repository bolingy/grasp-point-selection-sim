import zipfile
import os
import shutil
import trimesh
import random
import glob
import subprocess
import click
import io
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import google.auth.transport.requests
import google.oauth2.service_account
import pickle
import time

home_path = '../Google_Scanned_Objects/'

if not os.path.exists(f"{home_path}extracted_meshes"):
    os.makedirs(f"{home_path}extracted_meshes")


def authenticate(service_account_path='kernel/dynamo-grasp-sf-key.json'):
    """Authenticate using a service account or fall back to OAuth2."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # Try service account authentication first
    if os.path.exists(service_account_path):
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            service_account_path, scopes=SCOPES)
        print("Authenticated using service account.")
    else:
        # Fall back to OAuth2 authentication
        creds = None
        if os.path.exists('kernel/token.pickle'):
            with open('kernel/token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                'kernel/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('kernel/token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        credentials = creds
        print("Authenticated using OAuth2.")

    return build('drive', 'v3', credentials=credentials)


def download_folder(service, folder_id, output_folder):
    """Download all files from a specified Google Drive folder."""
    page_token = None
    all_files = []

    # Query the files in the folder with pagination
    while True:
        results = service.files().list(q=f"'{folder_id}' in parents",
                                       pageSize=1000,
                                       pageToken=page_token).execute()
        items = results.get('files', [])
        all_files.extend(items)
        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break

    # Check if there are files in the folder
    if not all_files:
        print("No files found in the folder.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        for item in all_files:
            download_file_with_retry(
                service, item['id'], os.path.join(output_folder, item['name']))


def download_file_with_retry(service, file_id, output_filename, retries=3, delay=5):
    """Attempt to download a file with retries in case of an error."""
    for attempt in range(retries):
        try:
            request = service.files().get_media(fileId=file_id)
            with io.FileIO(output_filename, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    print(
                        f"Downloaded {int(status.progress() * 100)}% of {output_filename}.")
            return  # Exit function after successful download
        except HttpError as error:
            print(
                f"Error encountered: {error}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"Failed to download {file_id} after {retries} attempts.")


def download_file(service, file_id, output_filename):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(output_filename, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(
                f"Downloaded {int(status.progress() * 100)}% of {output_filename}")


def delete_all_contents_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


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


def add_texture_to_mtl(mtl_filepath, texture_filename="texture.png"):
    shutil.move(f"{home_path}extracted_meshes/temp/materials/textures/texture.png",
                f"{home_path}extracted_meshes/temp/meshes/texture.png")
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
    "3F": [0.1, 0.2],
    "3E": [0.075, 0.12],
    "3H": [0.075, 0.13],
}


@click.command()
@click.option('--bin-id', type=click.Choice(['3H', '3E', '3F']), default='3F')
@click.option('--num-envs', default=50)
def main(bin_id, num_envs):
    files = [f for f in os.listdir(f"{home_path}") if os.path.isfile(
        os.path.join(f"{home_path}", f))]
    if len(files) < 300:
        service = authenticate(
            service_account_path='kernel/dynamo-grasp-sf-key.json')
        download_folder(
            service, "1l9fd_i1NYM0V0xx08hQvxVIVr0GIYZ92", home_path)

    while True:
        delete_all_contents_in_directory(
            'assets/google_scanned_models/')

        num_samples = 30
        # List all files with the specified extension
        files = glob.glob(os.path.join(f'{home_path}', '*.zip'))
        # Randomly sample files
        sampled_files = random.sample(files, num_samples)

        for name_of_file in sampled_files:
            # getting rid of path and '.zip' extension string
            name_of_file = os.path.basename(name_of_file)
            name_of_file = name_of_file[:-4]
            folder_path = f"assets/google_scanned_models/{name_of_file}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            folder_path = f"{home_path}extracted_meshes/temp"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            unzip_file(f"{home_path}{name_of_file}.zip",
                       f"{home_path}extracted_meshes/temp")

            length_bounds = random.uniform(
                bin_id_resize_bounds[bin_id][0], bin_id_resize_bounds[bin_id][1])
            width_bounds = random.uniform(
                bin_id_resize_bounds[bin_id][0], bin_id_resize_bounds[bin_id][1])
            height_bounds = random.uniform(
                bin_id_resize_bounds[bin_id][0], bin_id_resize_bounds[bin_id][1])
            scale_mesh_within_bounds(f"{home_path}extracted_meshes/temp/meshes/model.obj",
                                     f"{home_path}extracted_meshes/temp/meshes/resized_model.obj", length_bounds, width_bounds, height_bounds)

            mass = random.uniform(0.2, 1)

            inertia_tensor_scaled = get_mesh_inertia(
                f"{home_path}extracted_meshes/temp/meshes/resized_model.obj", mass)
            add_texture_to_mtl(
                f"{home_path}extracted_meshes/temp/meshes/material.mtl")
            create_urdf_with_inertia(f"{home_path}extracted_meshes/temp/meshes/model.urdf", mass,
                                     inertia_tensor_scaled[0, 0], inertia_tensor_scaled[1, 1], inertia_tensor_scaled[2, 2])
            shutil.move(f"{home_path}extracted_meshes/temp/meshes/texture.png",
                        f"assets/google_scanned_models/{name_of_file}/texture.png")
            shutil.move(f"{home_path}extracted_meshes/temp/meshes/material.mtl",
                        f"assets/google_scanned_models/{name_of_file}/material.mtl")
            shutil.move(f"{home_path}extracted_meshes/temp/meshes/resized_model.obj",
                        f"assets/google_scanned_models/{name_of_file}/resized_model.obj")
            shutil.move(f"{home_path}extracted_meshes/temp/meshes/model.urdf",
                        f"assets/google_scanned_models/{name_of_file}/model.urdf")

            delete_all_contents_in_directory(
                f"{home_path}extracted_meshes/temp")
            shutil.rmtree(f"{home_path}extracted_meshes/temp")

        delete_all_contents_in_directory(
            f"{home_path}extracted_meshes/")

        command = ["python", "data_collection.py", "--bin-id",
                   f"{bin_id}", "--num-envs", f"{num_envs}"]
        result = subprocess.run(command)

        if result.returncode == 0:
            print("Simulation exited successfully!")
        else:
            print(
                f"Simulation exit failed with return code {result.returncode}.")


if __name__ == "__main__":
    main()
