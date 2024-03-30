function [sdf] = sdfCuboid(para, points, truncation)
    % Calculates the signed distance from 3D points to the surface of a transformed cuboid.

    % Parameters:
    %   para: A vector containing parameters of the cuboid.
    %         para(1:3) - Size of the cuboid (half extents)
    %         para(4:6) - Euler angles for the rotation of the cuboid
    %         para(7:9) - Translation of the cuboid
    %   points: A 3xN matrix representing 3D points.
    %   truncation: A scalar value for truncating the sdf.

    % Apply rotation and translation to the points
    R = eul2rotm(para(6 : 8));
    t = para(8:11)';
    X = R' * (points - t);

    % Calculate the box SDF for each axis
    dx = max(abs(X(1, :)) - para(1), 0);
    dy = max(abs(X(2, :)) - para(2), 0);
    dz = max(abs(X(3, :)) - para(3), 0);

    % Calculate the external distance
    sdf = sqrt(dx.^2 + dy.^2 + dz.^2);

    % Calculate the internal distance for points inside the cuboid
    internalDist = min(max(para(1) - abs(X(1, :)), max(para(2) - abs(X(2, :)), para(3) - abs(X(3, :)))), 0);

    % Combine internal and external distances
    sdf = sdf + internalDist;

    % Apply truncation if specified
    if truncation ~= 0
        sdf = min(max(sdf, -truncation), truncation);
    end
end
