function compute_diagonals(escalings, materials, sx, sy, sz, lix_mat, liy_mat, liz_mat, lix_border, liy_border, liz_border)
    escaling_R = escalings["R"]
    escaling_Cd = escalings["Cd"]
    escaling_Lp = escalings["Lp"]
    eps0 = 8.854187816997944e-12
    for cont in range(1,length(materials))
        sigmar = materials[cont]["sigmar"]
        epsr = materials[cont]["epsr"]
        if sigmar != 0
            materials[cont]["Rx"] = 0.5 * sx / (sigmar * sy * sz)
            materials[cont]["Ry"] = 0.5 * sy / (sigmar * sx * sz)
            materials[cont]["Rz"] = 0.5 * sz / (sigmar * sy * sx)
            if epsr == 1
                materials[cont]["Cx"] = 0
                materials[cont]["Cy"] = 0
                materials[cont]["Cz"] = 0
            else
                materials[cont]["Cx"] = eps0 * (epsr - 1) * sy * sz / (0.5 * sx)
                materials[cont]["Cy"] = eps0 * (epsr - 1) * sx * sz / (0.5 * sy)
                materials[cont]["Cz"] = eps0 * (epsr - 1) * sy * sx / (0.5 * sz)
            end
        else
            epsr = materials[cont]["epsr"]
            materials[cont]["Rx"] = 0
            materials[cont]["Ry"] = 0
            materials[cont]["Rz"] = 0
            materials[cont]["Cx"] = eps0 * (epsr - 1) * sy * sz / (0.5 * sx)
            materials[cont]["Cy"] = eps0 * (epsr - 1) * sx * sz / (0.5 * sy)
            materials[cont]["Cz"] = eps0 * (epsr - 1) * sy * sx / (0.5 * sz)
        end
    end
    Rx = zeros(size(lix_border,1), 4)
    Ry = zeros(size(liy_border,1), 4)
    Rz = zeros(size(liz_border,1), 4)
    Cx = zeros(Complex64, size(lix_border,1), 4)
    Cy = zeros(Complex64, size(liy_border,1), 4)
    Cz = zeros(Complex64, size(liz_border,1), 4)
    for cont in range(1,length(materials))
        if materials[cont]["Rx"] != 0
            ind_m = findall(x -> x == cont, lix_mat[:, 1])
            Rx[ind_m, 1] .= materials[cont]["Rx"]
            ind_m = findall(x -> x == cont, lix_mat[:, 2])
            Rx[ind_m, 2] .= materials[cont]["Rx"]
            ind_m = findall(x -> x == cont, lix_border[:, 1])
            Rx[ind_m, 3] .= materials[cont]["Rx"]
            ind_m = findall(x -> x == cont, lix_border[:, 2])
            Rx[ind_m, 4] .= +materials[cont]["Rx"]
        end
        if materials[cont]["Cx"] != 0
            ind_m = findall(x -> x == cont, lix_mat[:, 1])
            Cx[ind_m, 1] .= materials[cont]["Cx"]
            ind_m = findall(x -> x == cont, lix_mat[:, 2])
            Cx[ind_m, 2] .= materials[cont]["Cx"]
            ind_m = findall(x -> x == cont, lix_border[:, 1])
            Cx[ind_m, 3] .= materials[cont]["Cx"]
            ind_m = findall(x -> x == cont, lix_border[:, 2])
            Cx[ind_m, 4] .= +materials[cont]["Cx"]
        end
        if materials[cont]["Ry"] != 0
            ind_m = findall(x -> x == cont, liy_mat[:, 1])
            Ry[ind_m, 1] .= materials[cont]["Ry"]
            ind_m = findall(x -> x == cont, liy_mat[:, 2])
            Ry[ind_m, 2] .= materials[cont]["Ry"]
            ind_m = findall(x -> x == cont, liy_border[:, 1])
            Ry[ind_m, 3] .= materials[cont]["Ry"]
            ind_m = findall(x -> x == cont, liy_border[:, 2])
            Ry[ind_m, 4] .= +materials[cont]["Ry"]
        end
        if materials[cont]["Cy"] != 0
            ind_m = findall(x -> x == cont, liy_mat[:, 1])
            Cy[ind_m, 1] .= materials[cont]["Cy"]
            ind_m = findall(x -> x == cont, liy_mat[:, 2])
            Cy[ind_m, 2] .= materials[cont]["Cy"]
            ind_m = findall(x -> x == cont, liy_border[:, 1])
            Cy[ind_m, 3] .= materials[cont]["Cy"]
            ind_m = findall(x -> x == cont, liy_border[:, 2])
            Cy[ind_m, 4] .= +materials[cont]["Cy"]
        end
        if materials[cont]["Rz"] != 0
            ind_m = findall(x -> x == cont, liz_mat[:, 1])
            Rz[ind_m, 1] .= materials[cont]["Rz"]
            ind_m = findall(x -> x == cont, liz_mat[:, 2])
            Rz[ind_m, 2] .= materials[cont]["Rz"]
            ind_m = findall(x -> x == cont, liz_border[:, 1])
            Rz[ind_m, 3] .= materials[cont]["Rz"]
            ind_m = findall(x -> x == cont, liz_border[:, 2])
            Rz[ind_m, 4] .= +materials[cont]["Rz"]
        end
        if materials[cont]["Cz"] != 0
            ind_m = findall(x -> x == cont, liz_mat[:, 1])
            Cz[ind_m, 1] .= materials[cont]["Cz"]
            ind_m = findall(x -> x == cont, liz_mat[:, 2])
            Cz[ind_m, 2] .= materials[cont]["Cz"]
            ind_m = findall(x -> x == cont, liz_border[:, 1])
            Cz[ind_m, 3] .= materials[cont]["Cz"]
            ind_m = findall(x -> x == cont, liz_border[:, 2])
            Cz[ind_m, 4] .= +materials[cont]["Cz"]
        end
    end
    lix_aux = ceil.(Int, lix_border[:, 1] / 100) + ceil.(Int, lix_border[:, 2] / 100)
    liy_aux = ceil.(Int, liy_border[:, 1] / 100) + ceil.(Int, liy_border[:, 2] / 100)
    liz_aux = ceil.(Int, liz_border[:, 1] / 100) + ceil.(Int, liz_border[:, 2] / 100)
    i1x = findall(x -> x == 1, lix_aux)
    i2x = findall(x -> x == 2, lix_aux)
    i1y = findall(x -> x == 1, liy_aux)
    i2y = findall(x -> x == 2, liy_aux)
    i1z = findall(x -> x == 1, liz_aux)
    i2z = findall(x -> x == 2, liz_aux)
    Self_x = Lp_self(sx, sy, sz)
    Self_y = Lp_self(sy, sx, sz)
    Self_z = Lp_self(sz, sy, sx)
    Self_x1 = Lp_self(sx + sx / 2, sy, sz)
    Self_y1 = Lp_self(sy + sy / 2, sx, sz)
    Self_z1 = Lp_self(sz + sz / 2, sy, sx)
    Self_x2 = Lp_self(2 * sx, sy, sz)
    Self_y2 = Lp_self(2 * sy, sx, sz)
    Self_z2 = Lp_self(2 * sz, sy, sx)
    diag_Lp_x = Self_x * ones(size(lix_aux, 1), 1)
    diag_Lp_y = Self_y * ones(size(liy_aux, 1), 1)
    diag_Lp_z = Self_z * ones(size(liz_aux, 1), 1)
    diag_Lp_x[i1x] = Self_x1 * ones(length(i1x), 1)
    diag_Lp_y[i1y] = Self_y1 * ones(length(i1y), 1)
    diag_Lp_z[i1z] = Self_z1 * ones(length(i1z), 1)
    diag_Lp_x[i2x] = Self_x2 * ones(length(i2x), 1)
    diag_Lp_y[i2y] = Self_y2 * ones(length(i2y), 1)
    diag_Lp_z[i2z] = Self_z2 * ones(length(i2z), 1)
    diagonals = Dict()
    diagonals["R"] = escaling_R * [Rx; Ry; Rz]
    diagonals["Cd"] = escaling_Cd * [Cx; Cy; Cz]
    diagonals["Lp"] = escaling_Lp * [diag_Lp_x; diag_Lp_y; diag_Lp_z]
    diagonals["fc_Lp"] = escaling_Lp * ([diag_Lp_x; diag_Lp_y; diag_Lp_z] - [Self_x * ones(size(lix_aux, 1), 1); Self_y * ones(size(liy_aux, 1), 1); Self_z * ones(size(liz_aux, 1), 1)])

    return diagonals
end

function Lp_self(l, W, T)
    w = W / l
    t = T / l
    r = sqrt(w^2 + t^2)
    aw = sqrt(w^2 + 1)
    at = sqrt(t^2 + 1)
    ar = sqrt(w^2 + t^2 + 1)
    mu0 = 4 * pi * 1e-7
    Lp_Self_Rect = 2 * mu0 * l / pi * (1 / 4 * (1 / w * asinh(w / at) + 1 / t * asinh(t / aw) + asinh(1 / r)) +
        1 / 24 * (t^2 / w * asinh(w / (t * at * (r + ar))) + w^2 / t * asinh(t / (w * aw * (r + ar))) +
        t^2 / w^2 * asinh(w^2 / (t * r * (at + ar))) + w^2 / t^2 * asinh(t^2 / (w * r * (aw + ar))) +
        1 / (w * t^2) * asinh(w * t^2 / (at * (aw + ar))) + 1 / (t * w^2) * asinh(t * w^2 / (aw * (at + ar)))) -
        1 / 6 * (1 / (w * t) * atan(w * t / ar) + t / w * atan(w / (t * ar)) + w / t * atan(t / (w * ar))) -
        1 / 60 * ((ar + r + t + at) * t^2 / ((ar + r) * (r + t) * (t + at) * (at + ar)) +
        (ar + r + w + aw) * w^2 / ((ar + r) * (r + w) * (w + aw) * (aw + ar)) +
        (ar + aw + 1 + at) / ((ar + aw) * (aw + 1) * (1 + at) * (at + ar))) -
        1 / 20 * (1 / (r + ar) + 1 / (aw + ar) + 1 / (at + ar)))

    return Lp_Self_Rect
end
