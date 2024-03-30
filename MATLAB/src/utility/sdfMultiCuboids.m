function [sdf_u] = sdfMultiCuboids(x, points, truncation)

    n = size(x, 1);
    sdf_u = sdfCuboid(x(1, :), points, truncation);
    if n > 1
        for i = 2 : n
            sdf_u = min(sdf_u, ...
                sdfCuboid(x(i, :), points, truncation));
            
        end
    end
    
    end