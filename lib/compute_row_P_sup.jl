function compute_row_P_sup(centers, centers2, sx, sy, sz, caser, caser2)
    eps0 = 8.854187816997944e-12
    if caser == 1
        A1 = sx * sz
        xc1 = centers[1]
        yc1 = centers[2] - sy / 2
        zc1 = centers[3]
        a1 = sx
        b1 = 0
        c1 = sz
    elseif caser == 2
        A1 = sx * sz
        xc1 = centers[1]
        yc1 = centers[2] + sy / 2
        zc1 = centers[3]
        a1 = sx
        b1 = 0
        c1 = sz
    elseif caser == 3
        A1 = sy * sz
        xc1 = centers[1] - sx / 2
        yc1 = centers[2]
        zc1 = centers[3]
        a1 = 0
        b1 = sy
        c1 = sz
    elseif caser == 4
        A1 = sy * sz
        xc1 = centers[1] + sx / 2
        yc1 = centers[2]
        zc1 = centers[3]
        a1 = 0
        b1 = sy
        c1 = sz
    elseif caser == 5
        A1 = sx * sy
        xc1 = centers[1]
        yc1 = centers[2]
        zc1 = centers[3] - sz / 2
        a1 = sx
        b1 = sy
        c1 = 0
    else
        A1 = sx * sy
        xc1 = centers[1]
        yc1 = centers[2]
        zc1 = centers[3] + sz / 2
        a1 = sx
        b1 = sy
        c1 = 0
    end
    N = size(centers2, 1)
    row = zeros(N, 1)
    if caser2 == 1
        for k = 1:N
            row[k] = 1 / (4 * pi * eps0 * A1 * sx * sz) * Integ_sup_sup(xc1, yc1, zc1, centers2[k, 1], centers2[k, 2] - sy / 2, centers2[k, 3], a1, b1, c1, sx, 0, sz)
        end
    elseif caser2 == 2
        for k = 1:N
            row[k] = 1 / (4 * pi * eps0 * A1 * sx * sz) * Integ_sup_sup(xc1, yc1, zc1, centers2[k, 1], centers2[k, 2] + sy / 2, centers2[k, 3], a1, b1, c1, sx, 0, sz)
        end
    elseif caser2 == 3
        for k = 1:N
            row[k] = 1 / (4 * pi * eps0 * A1 * sy * sz) * Integ_sup_sup(xc1, yc1, zc1, centers2[k, 1] - sx / 2, centers2[k, 2], centers2[k, 3], a1, b1, c1, 0, sy, sz)
        end
    elseif caser2 == 4
        for k = 1:N
            row[k] = 1 / (4 * pi * eps0 * A1 * sy * sz) * Integ_sup_sup(xc1, yc1, zc1, centers2[k, 1] + sx / 2, centers2[k, 2], centers2[k, 3], a1, b1, c1, 0, sy, sz)
        end
    elseif caser2 == 5
        for k = 1:N
            row[k] = 1 / (4 * pi * eps0 * A1 * sx * sy) * Integ_sup_sup(xc1, yc1, zc1, centers2[k, 1], centers2[k, 2], centers2[k, 3] - sz / 2, a1, b1, c1, sx, sy, 0)
        end
    else
        for k = 1:N
            row[k] = 1 / (4 * pi * eps0 * A1 * sx * sy) * Integ_sup_sup(xc1, yc1, zc1, centers2[k, 1], centers2[k, 2], centers2[k, 3] + sz / 2, a1, b1, c1, sx, sy, 0)
        end
    end
    return row
end

function Integ_sup_sup(xc1, yc1, zc1, xc2, yc2, zc2, a1, b1, c1, a2, b2, c2)
    use_suppression=1;

    epsilon1=5e-3;
    epsilon2=1e-3;
    epsilon3=1e-3;
    epsilon4=3e-1;

    x1v=[xc1-a1/2 xc1+a1/2];
    y1v=[yc1-b1/2 yc1+b1/2];
    z1v=[zc1-c1/2 zc1+c1/2];

    x2v=[xc2-a2/2 xc2+a2/2];
    y2v=[yc2-b2/2 yc2+b2/2];
    z2v=[zc2-c2/2 zc2+c2/2];

    sup1_xz_plane = 0;
    sup1_yz_plane = 0;

    sup2_xz_plane = 0;
    sup2_yz_plane = 0;

    if (a1 <= b1 && a1 <= c1)
        sup1_yz_plane = 1;
        a1 = 1;
    elseif (b1 <= a1 && b1 <= c1)
        sup1_xz_plane = 1;
        b1 = 1;
    else
        c1 = 1;
    end

    if (a2 <= b2 && a2 <= c2)
        sup2_yz_plane = 1;
        a2 = 1;
    elseif (b2 <= a2 && b2 <= c2)
        sup2_xz_plane = 1;
        b2 = 1;
    else
        c2 = 1;
    end

    sup1=a1*b1*c1;
    sup2=a2*b2*c2;

    supp_x1=0; supp_y1=0; supp_z1=0;
    supp_x2=0; supp_y2=0; supp_z2=0;

    if use_suppression==1
        
        aux_x=abs.([x1v[1]-x2v[1], x1v[1]-x2v[end], x1v[end]-x2v[1], x1v[end]-x2v[end]]);
        aux_y=abs.([y1v[1]-y2v[1], y1v[1]-y2v[end], y1v[end]-y2v[1], y1v[end]-y2v[end]]);
        aux_z=abs.([z1v[1]-z2v[1], z1v[1]-z2v[end] ,z1v[end]-z2v[1], z1v[end]-z2v[end]]);
        
        min_R=sqrt(minimum(aux_x)^2+minimum(aux_y)^2+minimum(aux_z)^2);
        
        max_d=maximum(abs.(aux_x));
        supp_x1=check_condition_P(epsilon1,epsilon2,epsilon3,epsilon4,sup1,sup2,max_d,min_R,a1,b1,c1);
        supp_x2=check_condition_P(epsilon1,epsilon2,epsilon3,epsilon4,sup1,sup2,max_d,min_R,a2,b2,c2);
        
        max_d=maximum(abs.(aux_y));
        supp_y1=check_condition_P(epsilon1,epsilon2,epsilon3,epsilon4,sup1,sup2,max_d,min_R,b1,a1,c1);
        supp_y2=check_condition_P(epsilon1,epsilon2,epsilon3,epsilon4,sup1,sup2,max_d,min_R,b2,a2,c2);
        
        max_d=maximum(abs.(aux_z));
        supp_z1=check_condition_P(epsilon1,epsilon2,epsilon3,epsilon4,sup1,sup2,max_d,min_R,c1,a1,b1);
        supp_z2=check_condition_P(epsilon1,epsilon2,epsilon3,epsilon4,sup1,sup2,max_d,min_R,c2,a2,b2);
        
    end

    if (sup1_yz_plane==1)
        supp_x1 = 1;
    elseif (sup1_xz_plane==1)
        supp_y1 = 1;
    else
        supp_z1 = 1;
    end

    if (sup2_yz_plane==1)
        supp_x2 = 1;
    elseif (sup2_xz_plane==1)
        supp_y2 = 1;
    else
        supp_z2 = 1;
    end

    sum_supp=supp_x1+supp_y1+supp_z1+supp_x2+supp_y2+supp_z2;

    if sum_supp==6 #point-point to use
        
        integ=sup1*sup2/sqrt((xc1-xc2)^2+(yc1-yc2)^2+(zc1-zc2)^2);
        
        
    elseif sum_supp==5 #point-line to use
        
        is_point_v1=0;
        if(supp_x1+supp_y1+supp_z1==3)
            is_point_v1=1;
        end
        
        if is_point_v1==1
            if supp_x2==0 #line of volume 2 along x
                integ=sup1*sup2/a2*integ_line_point([x2v[1], x2v[end]],yc2,zc2, xc1,yc1,zc1);
            elseif supp_y2==0 #line of volume 2 along y
                integ=sup1*sup2/b2*integ_line_point([y2v[1], y2v[end]],xc2,zc2, yc1,xc1,zc1);
            else #line of volume 2 along z
                integ=sup1*sup2/c2*integ_line_point([z2v[1], z2v[end]],xc2,yc2, zc1,xc1,yc1);
            end
        else
            if supp_x1==0 #line of volume 1 along x
                integ=sup1*sup2/a1*integ_line_point([x1v[1], x1v[end]],yc1,zc1, xc2,yc2,zc2);
            elseif supp_y1==0 #line of volume 1 along y
                integ=sup1*sup2/b1*integ_line_point([y1v[1], y1v[end]],xc1,zc1, yc2,xc2,zc2);
            else #line of volume 1 along z
                integ=sup1*sup2/c1*integ_line_point([z1v[1], z1v[end]],xc1,yc1, zc2,xc2,yc2);
            end
        end
    elseif sum_supp==4 #point-surface or line-line case
        
        is_point_v1=0;
        if(supp_x1+supp_y1+supp_z1==3)
            is_point_v1=1;
        end
        is_point_v2=0;
        if(supp_x2+supp_y2+supp_z2==3)
            is_point_v2=1;
        end
        
        if is_point_v1==1 #point-surface case
            
            if supp_x2==1 #surface of volume 2 in yz plane
                integ=sup1*a2*integ_point_sup(zc1,yc1,xc1,[z2v[1], z2v[end]],[y2v[1], y2v[end]],xc2);
            elseif supp_y2==1 #surface of volume 2 in xz plane
                integ=sup1*b2*integ_point_sup(xc1,zc1,yc1,[x2v[1], x2v[end]],[z2v[1], z2v[end]],yc2);
            else #surface of volume 2 in xy plane
                integ=sup1*c2*integ_point_sup(xc1,yc1,zc1,[x2v[1], x2v[end]],[y2v[1], y2v[end]],zc2);
            end
        elseif is_point_v2==1 #point-surface case
            
            if supp_x1==1 #surface of volume 1 in yz plane
                integ=sup2*a1*integ_point_sup(zc2,yc2,xc2,[z1v[1], z1v[end]],[y1v[1], y1v[end]],xc1);
            elseif supp_y1==1 #surface of volume 1 in xz plane
                integ=sup2*b1*integ_point_sup(xc2,zc2,yc2,[x1v[1], x1v[end]],[z1v[1], z1v[end]],yc1);
            else #surface of volume 1 in xy plane
                integ=sup2*c1*integ_point_sup(xc2,yc2,zc2,[x1v[1], x1v[end]],[y1v[1], y1v[end]],zc1);
            end
        else #line-line case
            
            if supp_y1==1 && supp_z1==1
                if supp_y2==1 && supp_z2==1 # parallel lines
                    integ= b1*c1*b2*c2* integ_line_line_parall([x1v[1], x1v[end]],yc1,zc1,[x2v[1], x2v[end]],yc2,zc2);
                elseif supp_x2==1 && supp_z2==1 # orthogonal lines
                    integ= b1*c1*a2*c2*integ_line_line_ortho_xy([x1v[1], x1v[end]],yc1,zc1,xc2,[y2v[1], y2v[end]],zc2);
                else
                    integ= b1*c1*a2*b2*integ_line_line_ortho_xy([x1v[1], x1v[end]],zc1,yc1,xc2,[z2v[1], z2v[end]],yc2);
                end
            elseif supp_x1==1 && supp_z1==1
                
                if supp_x2==1 && supp_z2==1 # parallel lines
                    integ= a1*c1*a2*c2* integ_line_line_parall([y1v[1], y1v[end]],xc1,zc1,[y2v[1], y2v[end]],xc2,zc2);
                elseif supp_x2==1 && supp_y2==1 # orthogonal lines
                    integ= a1*c1*a2*b2*integ_line_line_ortho_xy([y1v[1], y1v[end]],zc1,xc1,yc2,[z2v[1], z2v[end]],xc2);
                else
                    integ= a1*c1*b2*c2*integ_line_line_ortho_xy([y1v[1], y1v[end]],xc1,zc1,yc2,[x2v[1], x2v[end]],zc2);
                end
            else
                if supp_x2==1 && supp_y2==1 # parallel lines
                    integ= a1*b1*a2*b2* integ_line_line_parall([z1v[1], z1v[end]],xc1,yc1,[z2v[1], z2v[end]],xc2,yc2);
                elseif supp_x2==1 && supp_z2==1 # orthogonal lines
                    integ= a1*b1*a2*c2*integ_line_line_ortho_xy([z1v[1], z1v[end]],yc1,xc1,zc2,[y2v[1], y2v[end]],xc2);
                else
                    integ= a1*b1*b2*c2*integ_line_line_ortho_xy([z1v[1], z1v[end]],xc1,yc1,zc2,[x2v[1], x2v[end]],yc2);
                end
            end
        end
    elseif sum_supp==3 #surface-line
        
        is_surf_v1=0;
        if(supp_x1+supp_y1+supp_z1==1)
            is_surf_v1=1;
        end
        
        #line-surface case
        if is_surf_v1==1 # bar1 is a surface
            
            if supp_x1==1# bar1 is a surface in y-z plane
                
                if supp_x2==0 #bar 2 is a line along x
                    integ=a1*b2*c2 * integ_line_surf_ortho([x2v[1], x2v[end]],yc2,zc2, xc1,[y1v[1], y1v[end]],[z1v[1], z1v[end]]);
                elseif supp_y2==0 #bar 2 is a line along y
                    integ=a1*a2*c2 *integ_line_surf_para( [y1v[1], y1v[end]],[z1v[1], z1v[end]],xc1,[y2v[1], y2v[end]],zc2,xc2);
                else #bar 2 is a line along z
                    integ=a1*a2*b2 *integ_line_surf_para( [z1v[1], z1v[end]],[y1v[1], y1v[end]],xc1,[z2v[1], z2v[end]],yc2,xc2);
                end
            elseif supp_y1==1# bar1 is a surface in x-z plane
                
                if supp_x2==0 #bar 2 is a line along x
                    integ=b1*b2*c2 * integ_line_surf_para( [x1v[1], x1v[end]],[z1v[1], z1v[end]],yc1,[x2v[1], x2v[end]],zc2,yc2);
                elseif supp_y2==0 #bar 2 is a line along y
                    integ=b1*a2*c2 *integ_line_surf_ortho([y2v[1], y2v[end]],xc2,zc2, yc1,[x1v[1], x1v[end]],[z1v[1], z1v[end]]);
                else #bar 2 is a line along z
                    integ=b1*a2*b2 *integ_line_surf_para( [z1v[1], z1v[end]],[x1v[1], x1v[end]],yc1,[z2v[1], z2v[end]],xc2,yc2);
                end
            else # bar1 is a surface in x-y plane
                
                if supp_x2==0 #bar 2 is a line along x
                    integ=c1*b2*c2 * integ_line_surf_para( [x1v[1], x1v[end]],[y1v[1], y1v[end]],zc1,[x2v[1], x2v[end]],yc2,zc2);
                elseif supp_y2==0 #bar 2 is a line along y
                    integ=c1*a2*c2 *integ_line_surf_para( [y1v[1], y1v[end]],[x1v[1], x1v[end]],zc1,[y2v[1], y2v[end]],xc2,zc2);
                else #bar 2 is a line along z
                    integ=c1*a2*b2 *integ_line_surf_ortho([z2v[1], z2v[end]],xc2,yc2, zc1,[x1v[1], x1v[end]],[y1v[1], y1v[end]]);
                end
            end
        else #bar2 is a surface
            if supp_x2==1# bar2 is a surface in y-z plane
                
                if supp_x1==0 #bar 1 is a line along x
                    integ=a2*b1*c1 * integ_line_surf_ortho([x1v[1], x1v[end]],yc1,zc1, xc2,[y2v[1], y2v[end]],[z2v[1], z2v[end]]);
                elseif supp_y1==0 #bar 1 is a line along y
                    integ=a2*a1*c1 *integ_line_surf_para( [y2v[1], y2v[end]],[z2v[1], z2v[end]],xc2,[y1v[1], y1v[end]],zc1,xc1);
                else #bar 1 is a line along z
                    integ=a2*a1*b1 *integ_line_surf_para( [z2v[1], z2v[end]],[y2v[1], y2v[end]],xc2,[z1v[1], z1v[end]],yc1,xc1);
                end
            elseif supp_y2==1# bar2 is a surface in x-z plane
                
                if supp_x1==0 #bar 1 is a line along x
                    integ=b2*b1*c1 * integ_line_surf_para( [x2v[1], x2v[end]],[z2v[1], z2v[end]],yc2,[x1v[1], x1v[end]],zc1,yc1);
                elseif supp_y1==0 #bar 1 is a line along y
                    integ=b2*a1*c1 *integ_line_surf_ortho([y1v[1], y1v[end]],xc1,zc1, yc2,[x2v[1], x2v[end]],[z2v[1], z2v[end]]);
                else #bar 1 is a line along z
                    integ=b2*a1*b1 *integ_line_surf_para( [z2v[1], z2v[end]],[x2v[1], x2v[end]],yc2,[z1v[1], z1v[end]],xc1,yc1);
                end
            else # bar2 is a surface in x-y plane
                
                if supp_x1==0 #bar 1 is a line along x
                    integ=c2*b1*c1 * integ_line_surf_para( [x2v[1], x2v[end]],[y2v[1], y2v[end]],zc2,[x1v[1], x1v[end]],yc1,zc1);
                elseif supp_y1==0 #bar 1 is a line along y
                    integ=c2*a1*c1 *integ_line_surf_para( [y2v[1], y2v[end]],[x2v[1], x2v[end]],zc2,[y1v[1], y1v[end]],xc1,zc1);
                else #bar 1 is a line along z
                    integ=c2*a1*b1 *integ_line_surf_ortho([z1v[1], z1v[end]],xc1,yc1, zc2,[x2v[1], x2v[end]],[y2v[1], y2v[end]]);
                end
            end
        end
    else #surface-surface case
        if supp_x1==1 #bar1 is a surface in yz plane
            
            if supp_x2==1 #bar2 is a surface in yz plane
                integ=a1*a2*integ_surf_surf_para([y1v[1], y1v[end]],[z1v[1], z1v[end]],xc1, [y2v[1], y2v[end]],[z2v[1], z2v[end]],xc2);
            elseif supp_y2==1 #bar2 is a surface in xz plane
                integ=a1*b2*integ_surf_surf_ortho([z1v[1], z1v[end]],[y1v[1], y1v[end]],xc1,[z2v[1], z2v[end]],yc2,[x2v[1], x2v[end]]);
            else #bar2 is a surface in xy plane
                integ=a1*c2*integ_surf_surf_ortho([y1v[1], y1v[end]],[z1v[1], z1v[end]],xc1,[y2v[1], y2v[end]],zc2,[x2v[1], x2v[end]]);
            end
        elseif supp_y1==1 #bar1 is a surface in xz plane
            
            if supp_x2==1 #bar2 is a surface in yz plane
                integ=b1*a2*integ_surf_surf_ortho([z1v[1], z1v[end]],[x1v[1], x1v[end]],yc1,[z2v[1], z2v[end]],xc2,[y2v[1], y2v[end]]);
            elseif supp_y2==1 #bar2 is a surface in xz plane
                integ=b1*b2*integ_surf_surf_para([x1v[1], x1v[end]],[z1v[1], z1v[end]],yc1, [x2v[1], x2v[end]],[z2v[1], z2v[end]],yc2);
            else #bar2 is a surface in xy plane
                integ=b1*c2*integ_surf_surf_ortho([x1v[1], x1v[end]],[z1v[1], z1v[end]],yc1,[x2v[1], x2v[end]],zc2,[y2v[1], y2v[end]]);
            end
        else #bar1 is a surface in xy plane
            
            if supp_x2==1 #bar2 is a surface in yz plane
                integ=c1*a2*integ_surf_surf_ortho([y1v[1], y1v[end]],[x1v[1], x1v[end]],zc1,[y2v[1], y2v[end]],xc2,[z2v[1], z2v[end]]);
            elseif supp_y2==1 #bar2 is a surface in xz plane
                integ=c1*b2*integ_surf_surf_ortho([x1v[1], x1v[end]],[y1v[1], y1v[end]],zc1,[x2v[1], x2v[end]],yc2,[z2v[1], z2v[end]]);
            else #bar2 is a surface in xy plane
                integ=c1*c2*integ_surf_surf_para([x1v[1], x1v[end]],[y1v[1], y1v[end]],zc1, [x2v[1], x2v[end]],[y2v[1], y2v[end]],zc2);
            end
        end
        
    end
    return integ
end

function check_condition_P(eps1, eps2, eps3, eps4, Sup1, Sup2, max_d, min_R, size_dim, other_dim1, other_dim2)
    max_oth = other_dim1 * other_dim2
    condS = max(max_oth, size_dim) / (min_R .+ 1e-15)
    condX1a = Sup1 * Sup2 * max_d / (min_R .+ 1e-15).^3
    condX1f = size_dim / (min_R .+ 1e-15)
    condX1b = size_dim / max_oth
    supp_dim = 0
    if all(condS .< eps4) && (all(condX1b .<= eps3) || all(condX1f .< eps1)) && all(condX1a .< eps2)
        supp_dim = 1
    end
    return supp_dim
end

function integ_line_point(x1v, y3, z3, x1, y1, z1)
    x3 = x1v[1]
    x4 = x1v[end]
    check = 1 / (x1 - x3)
    if isnan(check) || isinf(check)
        x1 = x1 - 1e-8
    end
    check = 1 / (x1 - x4)
    if isnan(check) || isinf(check) && x1 > 0
        x1 = x1 + 1e-8
    end
    R1 = sqrt((x1 - x3)^2 + (y1 - y3)^2 + (z1 - z3)^2)
    R2 = sqrt((x1 - x4)^2 + (y1 - y3)^2 + (z1 - z3)^2)
    Ip = (log((x1 - x3) + R1)) - (log((x1 - x4) + R2))
    if isnan(Ip) || isinf(Ip)
        Ip = (x1 - x3) / abs((x1 - x3)) * log(abs(x1 - x3)) - (x1 - x4) / abs((x1 - x4)) * log(abs(x1 - x4))
    end
    return Ip
end

function integ_point_sup(x1, y1, z1, x2v, y2v, z2)
    sol = 0.0
    for c1 = 1:2
        x2 = x2v[c1]
        for c2 = 1:2
            y2 = y2v[c2]
            R = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
            term1 = (x1 - x2) * real(log(complex((y1 - y2) + R)))
            if isnan(term1) || isinf(term1)
                term1 = 0.0
            end
            term2 = (y1 - y2) * real(log(complex((x1 - x2) + R)))
            if isnan(term2) || isinf(term2)
                term2 = 0.0
            end
            term3 = -abs(z1 - z2) * atan((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))
            sol = sol + (-1)^(c1 + c2) * (term1 + term2 + term3)
        end
    end
    return sol
end

function integ_line_line_parall(x1v, y1, z1, x2v, y2, z2)
    dy = (y1 - y2)
    dz = (z1 - z2)
    if abs(dy) < 1e-10 && abs(dz) < 1e-10
        sol = integ_line_line_sp(x1v, x2v)
    else
        sol = 0.0
        for c1 = 1:2
            x1 = x1v[c1]
            for c2 = 1:2
                x2 = x2v[c2]
                R = sqrt((x1 - x2)^2 + dy^2 + dz^2)
                term1 = (x1 - x2) * real(log(complex((x1 - x2) + R)))
                if isnan(term1) || isinf(term1)
                    term1 = 0.0
                end
                sol = sol + (-1)^(c1 + c2 + 1) * (term1 - R)
            end
        end
    end
    return sol
end

function integ_line_line_sp(x1v, x2v)
    sol = 0.0
    for c1 = 1:2
        x1 = x1v[c1]
        for c2 = 1:2
            x2 = x2v[c2]
            R = sqrt((x1 - x2)^2)
            term1 = (x1 - x2) / R * (x1 - x2 * real(log(complex((x2 - x1))))) + x1 * real(log(complex((x1 - x2))))
            if isnan(term1) || isinf(term1)
                term1 = 0.0
            end
            sol = sol + (-1)^(c1 + c2 + 1) * (term1)
        end
    end
    return sol
end

function integ_line_line_ortho_xy(x1v, y1, z1, x2, y2v, z2)
    sol = 0.0
    for c1 = 1:2
        x1 = x1v[c1]
        for c2 = 1:2
            y2 = y2v[c2]
            R = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
            term1 = (x1 - x2) * real(log(complex((y1 - y2) + R)))
            if isnan(term1) || isinf(term1)
                term1 = 0.0
            end
            term2 = (y1 - y2) * real(log(complex((x1 - x2) + R)))
            if isnan(term2) || isinf(term2)
                term2 = 0.0
            end
            term3 = -abs(z1 - z2) * atan((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))
            sol = sol + (-1)^(c1 + c2 + 1) * (term1 + term2 + term3)
        end
    end
    return sol
end

function integ_line_surf_ortho(x1v, y1, z1, x2, y2v, z2v)
    sol = 0.0
    for c1 = 1:2
        x1 = x1v[c1]
        for c2 = 1:2
            y2 = y2v[c2]
            for c3 = 1:2
                z2 = z2v[c3]
                R = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
                term1 = (y1 - y2) * (z1 - z2) * real(log(complex((x1 - x2) + R)))
                if isnan(term1) || isinf(term1)
                    term1 = 0.0
                end
                term2 = (x1 - x2) * (z1 - z2) * real(log(complex((y1 - y2) + R)))
                if isnan(term2) || isinf(term2)
                    term2 = 0.0
                end
                term3 = (y1 - y2) * (x1 - x2) * real(log(complex((z1 - z2) + R)))
                if isnan(term3) || isinf(term3)
                    term3 = 0.0
                end
                term4 = -1/2 * abs(z1 - z2) * (z1 - z2) * atan((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))
                term5 = -1/2 * abs(y1 - y2) * (y1 - y2) * atan((x1 - x2) * (z1 - z2), (abs(y1 - y2) * R))
                term6 = -1/2 * abs(x1 - x2) * (x1 - x2) * atan((y1 - y2) * (z1 - z2), (abs(x1 - x2) * R))
                sol = sol + (-1)^(c1 + c2 + c3) * (term1 + term2 + term3 + term4 + term5 + term6)
            end
        end
    end
    return sol
end

function integ_line_surf_para(x1v, y1v, z1, x2v, y2, z2)
    sol = 0.0
    for c1 = 1:2
        x1 = x1v[c1]
        for c2 = 1:2
            y1 = y1v[c2]
            for c3 = 1:2
                x2 = x2v[c3]
                R = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
                term1 = (x1 - x2) * (y1 - y2) * real(log(complex((x1 - x2) + R)))
                if isnan(term1) || isinf(term1)
                    term1 = 0.0
                end
                term2 = ((x1 - x2)^2 - (z1 - z2)^2) / 2 * real(log(complex((y1 - y2) + R)))
                if isnan(term2) || isinf(term2)
                    term2 = 0.0
                end
                term3 = -(x1 - x2) * abs(z1 - z2) * atan((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))
                term4 = -(y1 - y2) / 2 * R
                sol = sol + (-1)^(c1 + c2 + c3 + 1) * (term1 + term2 + term3 + term4)
            end
        end
    end
    return sol
end

function integ_surf_surf_ortho(x1v, y1v, z1, x2v, y2, z2v)
    sol = 0.0
    for c1 in 1:2
        x1 = x1v[c1]
        for c2 in 1:2
            y1 = y1v[c2]
            for c3 in 1:2
                z2 = z2v[c3]
                for c4 in 1:2
                    x2 = x2v[c4]
                    R = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
                    term1 = -1/3 * (y1-y2) * (z1-z2) * R
                    term2 = -1/2 * (x1-x2) * abs(z1-z2) * (z1-z2) * atan((x1-x2) * (y1-y2), abs(z1-z2) * R)
                    term3 = -1/2 * (x1-x2) * abs(y1-y2) * (y1-y2) * atan((x1-x2) * (z1-z2), abs(y1-y2) * R)
                    term4 = -1/6 * abs(x1-x2)^3 * atan((y1-y2) * (z1-z2), abs(x1-x2) * R)
                    term5 = (x1-x2) * (y1-y2) * (z1-z2) * real(log(complex((x1-x2) + R)))
                    if isnan(term5) || isinf(term5)
                        term5 = 0.0
                    end
                    term6 = (1/2 * (x1-x2)^2 - 1/6 * (z1-z2)^2) * (z1-z2) * real(log(complex((y1-y2) + R)))
                    if isnan(term6) || isinf(term6)
                        term6 = 0.0
                    end
                    term7 = (1/2 * (x1-x2)^2 - 1/6 * (y1-y2)^2) * (y1-y2) * real(log(complex((z1-z2) + R)))
                    if isnan(term7) || isinf(term7)
                        term7 = 0.0
                    end
                    sol = sol + (-1)^(c1+c2+c3+c4) * (term1+term2+term3+term4+term5+term6+term7)
                end
            end
        end
    end
    return sol
end
