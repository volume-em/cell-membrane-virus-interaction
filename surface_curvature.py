import os
import yaml
import argparse
import trimesh
import numpy as np
import pandas as pd
import open3d as o3d
import seaborn as sns
from scipy.spatial import cKDTree
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from skimage.transform import resize
from scipy.stats import ttest_1samp
from trimesh.curvature import discrete_mean_curvature_measure as mean_curvature

def markups_to_numpy(
    markups_fpath, 
    deduplicate=True, 
    min_separation=50, 
    lps_to_ras=False
):
    """
    Convert from Slicer markups .fcsv text file to numpy array of xyz coordinates.
    """
    
    with open(markups_fpath, mode='r') as f:
        lines = f.readlines()[3:] # first 3 are header

    # xyz strings to floats
    xyz_points = np.array([list(map(lambda x: float(x), l.split(',')[1:4])) for l in lines])

    # removes points that are nearer than
    # min separation from each other
    if deduplicate:
        deduped_points = []
        remaining_points = np.copy(xyz_points)
        while len(remaining_points) > 0:
            cdists = np.linalg.norm(remaining_points[:1] - remaining_points, axis=1)
            deduped_points.append(remaining_points[0])
            remove_indices = np.where(cdists < min_separation)[0]
            remaining_points = np.delete(remaining_points, remove_indices, axis=0)

        xyz_points = np.array(deduped_points)
        
    # markups may be in LPS, Slicer visualization requires RAS
    # conversion; just invert x and y coordinates
    if lps_to_ras:
        xyz_points[:, 0] *= -1
        xyz_points[:, 1] *= -1
        
    return xyz_points

def numpy_to_markups(markups_fpath, array):
    """
    Converts from a numpy array of xyz coordinates to a Slicer
    markups .fcsv file.
    """
    with open(markups_fpath, mode='r') as f:
        in_lines = f.readlines() # first 4 are headers
        
    # copy header
    out_lines = in_lines[:3]
    template_line = in_lines[4].split(',')
    
    for ix, point in enumerate(array):
        x, y, z = point
        new_line = []
        new_line.append(f'vtkMRMLMarkupsFiducialNode_{ix}')
        new_line.extend([str(x), str(y), str(z)])
        new_line.extend(template_line[4:11])
        new_line.append(f'V-{ix + 1}')
        new_line.extend(template_line[12:])
        
        out_lines.append(','.join(new_line))
        
    return out_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help="Path to the .yaml config file")
    args = parser.parse_args()
    
    # load the config file
    with open(args.config_file, mode='r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
        
    # read in the parameters
    wdir = config['wdir']
    mesh_fpath = os.path.join(wdir, config['mesh_fname'])
    ref_mesh_fpath = os.path.join(wdir, config['ref_mesh_fname'])
    markups_fpath = os.path.join(wdir, config['markups_fname'])
    reference_is_resin = config['reference_is_resin']
    dedupe = config['dedupe']
    min_separation = config['min_separation']
    lps_to_ras = config['lps_to_ras']
    curve_r = config['curvature_radius']
    
    # load the meshes and virus centers
    mesh = trimesh.load(mesh_fpath)
    ref_mesh = trimesh.load(ref_mesh_fpath)
    virus_centers = markups_to_numpy(
        markups_fpath, deduplicate=dedupe, 
        min_separation=min_separation, lps_to_ras=lps_to_ras
    )
    print(f'Loaded mesh with {len(mesh.vertices)} and reference with {len(ref_mesh.vertices)} vertices')
    print(f'Loaded markups with {len(virus_centers)} virus fiducials')
    
    # calculate the mean curvature
    # and store is as an .npy array for later
    print(f'Calculating the mean curvature...')
    curvature_fpath = os.path.join(wdir, mesh_fpath.replace('.stl', f'_curvatures_r{curve_r}.npy'))
    if os.path.isfile(curvature_fpath):
        curvatures = np.load(curvature_fpath)
    else:
        curvatures = mean_curvature(mesh, points=mesh.vertices, radius=curve_r)
        np.save(curvature_fpath, curvatures)
        
    # create color mesh from curvature mapping
    # power gamma to spread the distribution and clipping
    # at the extrema to avoid washing out signal
    norm = PowerNorm(gamma=1.4, vmin=np.quantile(curvatures, 0.05), vmax=np.quantile(curvatures, 0.95))
    mapper = cm.ScalarMappable(norm=norm, cmap='jet')
    colors = mapper.to_rgba(curvatures)
    mesh.visual.vertex_colors = colors
    mesh.export(os.path.join(wdir, mesh_fpath.replace('.stl', f'_curvature_color_r{curve_r}.ply')))
    
    if reference_is_resin:
        sample_mesh = mesh
    else:
        sample_mesh = ref_mesh
    
    # determine which viruses are on the cell's surface
    max_distance = 100 # nm
    virus_distances = sample_mesh.nearest.on_surface(virus_centers)[1]
    viruses_on_surface = virus_centers[virus_distances <= max_distance]
    print(f'Found {len(viruses_on_surface)} viruses near surface of membrane.')
    
    # create open 3d mesh from which to sample points
    vertices = o3d.utility.Vector3dVector(sample_mesh.vertices)
    triangles = o3d.utility.Vector3iVector(sample_mesh.faces)
    tmesh = o3d.geometry.TriangleMesh(vertices, triangles)
    
    # sample 200K points and take 100K evenly spaced
    pcd = tmesh.sample_points_uniformly(number_of_points=200000)
    pcd = tmesh.sample_points_poisson_disk(number_of_points=100000, pcl=pcd)
    
    if reference_is_resin:
        pcd_points = np.asarray(pcd.points)
    else:
        pcd_points = mesh.nearest.on_surface(np.asarray(pcd.points))[0]
        
    # filter points on the boundary
    pcd_ref_distances = ref_mesh.nearest.vertex(pcd_points)[0]
    pcd_clear = pcd_points[pcd_ref_distances < 100]
    pcd_curvatures = mean_curvature(mesh, points=pcd_clear, radius=100)
    
    # get indices of points in sampled point cloud that are 100 nm
    # from virus' coordinates projected surface points
    pcd_kdt = cKDTree(pcd_clear)
    nearest_pcd_idx = pcd_kdt.query(viruses_on_surface)[1]
    near_virus_indices = np.concatenate(
        [idx for idx in pcd_kdt.query_ball_point(pcd_clear[nearest_pcd_idx], r=100)]
    )
    
    # determine set of curvatures near a virus and the population curvatures
    near_virus_curvatures = pcd_curvatures[np.unique(near_virus_indices)]
    population_curvatures = np.copy(pcd_curvatures)
    
    # plot the results
    plt.figure(figsize=(16, 8))
    sns.kdeplot(population_curvatures, color="red", shade=True, bw=20)
    sns.kdeplot(near_virus_curvatures, color="blue", shade=True, bw=20)
    plt.yticks([])
    plt.ylim(0, 0.005)
    plt.xticks(fontsize=20, fontname='Arial')
    plt.xlim(-1000, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, config['kde_fname']), dpi=300)
    
    # measure significance and save the csv
    curve_p = ttest_1samp(near_virus_curvatures, population_curvatures.mean()).pvalue
    df = pd.DataFrame({
        'location': ['membrane < 100 nm from virus', 'entire cell membrane'],
        'n_samples': [len(near_virus_curvatures), len(population_curvatures)],
        'curvature_r': [f'{curve_r} nm', f'{curve_r} nm'],
        'avg_mean_curvature': [near_virus_curvatures.mean(), population_curvatures.mean()],
        'median_mean_curvature': [np.median(near_virus_curvatures), np.median(population_curvatures)],
        'min_mean_curvature': [near_virus_curvatures.min(), population_curvatures.min()],
        'max_mean_curvature': [near_virus_curvatures.max(), population_curvatures.max()],
        'max_mean_curvature': [near_virus_curvatures.max(), population_curvatures.max()],
        'std_mean_curvature': [near_virus_curvatures.std(), population_curvatures.std()],
        'OneSample_ttest_pvalue': [curve_p, curve_p],
        'number_of_viruses': [len(viruses_on_surface), len(viruses_on_surface)]
      })

    df.to_csv(os.path.join(output_path, config['csv_fname']))