function []=run_mps(file_path, save_path)
    % clear
    % close all
    addpath('src/utility/')
    addpath('src/')
    addpath('../mesh2tri/')


    sdf = csvread(file_path)';
    voxelGrid.size = ones(1, 3) * sdf(1);
    voxelGrid.range = sdf(2 : 7);
    sdf = sdf(8 : end);

    voxelGrid.x = linspace(voxelGrid.range(1), voxelGrid.range(2), voxelGrid.size(1));
    voxelGrid.y = linspace(voxelGrid.range(3), voxelGrid.range(4), voxelGrid.size(2));
    voxelGrid.z = linspace(voxelGrid.range(5), voxelGrid.range(6), voxelGrid.size(3));
    [x, y, z] = ndgrid(voxelGrid.x, voxelGrid.y, voxelGrid.z);
    voxelGrid.points = reshape(cat(4, x, y, z), [], 3)';
    voxelGrid.interval = (voxelGrid.range(2) - voxelGrid.range(1)) / (voxelGrid.size(1) -1);
    voxelGrid.truncation = 1.2 * voxelGrid.interval; %1.2
    voxelGrid.disp_range = [-inf, voxelGrid.truncation];
    voxelGrid.visualizeArclength = 0.01 * sqrt(voxelGrid.range(2) - voxelGrid.range(1));
    clearvars x y z

    sdf = min(max(sdf, -voxelGrid.truncation), voxelGrid.truncation);

    %% marching-primitives
    my_tic = tic;
    [x] = MPS(sdf, voxelGrid);
    mps_time = toc(my_tic);

    save(save_path,"x", "mps_time")
end