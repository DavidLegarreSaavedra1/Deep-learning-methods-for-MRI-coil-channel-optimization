from methods.matrix_manipulation import *


def ROVir(coils, regions, lowf):
    A_W, A_H, B1_W, B1_H, B2_W = regions

    w = filter_coils(coils)
    tmp_A = w[A_H, A_W, :]
    img_A = tmp_A.flatten().reshape(tmp_A.shape[0]*tmp_A.shape[1], w.shape[2])

    tmp_B1 = w[B1_H, B1_W, :]
    tmp_B2 = w[B1_H, B2_W, :]

    img_B1 = tmp_B1.flatten().reshape(
        tmp_B1.shape[0]*tmp_B1.shape[1], w.shape[2])
    img_B2 = tmp_B2.flatten().reshape(
        tmp_B2.shape[0]*tmp_B2.shape[1], w.shape[2])

    img_B = np.append(img_B1, img_B2, axis=0)

    # tmp = img_B.reshape(
    #    tmp_B1.shape[0], tmp_B1.shape[1], 1
    # )
    A = generate_matrix(img_A)
    B = generate_matrix(img_B)

    # Calculate eigenvalues and eigenvectors both matrices
    general_matrix = LA.inv(B).dot(A)
    weights = calculate_eig(general_matrix, lowf)

    new_coils = coils[:, :, weights]
    new_coils_reshaped = new_coils.reshape(
        new_coils.shape[0]*new_coils.shape[1], new_coils.shape[2]
    )

    v_coils = generate_virtual_coils(new_coils_reshaped, weights)

    v_coils = v_coils.reshape(
        new_coils.shape[0], new_coils.shape[1], new_coils.shape[2]
    )

    return v_coils
