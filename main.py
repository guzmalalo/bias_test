import numpy as np
from bias_test import BiasTest
import matplotlib.pyplot as plt


def importa_data(input_file: str, delimiter=None):
    data = np.genfromtxt(input_file, skip_header=1 , delimiter=delimiter)

    # Offset displacement and force
    d_exp = data[:, 0] - data[0, 0]  # [mm]
    f_exp = data[:, 1] - data[0, 1]  # [N]

    return (d_exp, f_exp)

if __name__ == "__main__":
    # Experimental data
    d_exp, f_exp = importa_data("test_data/glass_plain_150_450.csv", delimiter=',')

    # Bias test object
    sample = BiasTest(
        displacement=d_exp,
        force=f_exp,
        width=150,
        length=450,
        thickness=1,
        material_name="Flax 250",
    )

    sample.plot_force_displacement()
    sample.plot_angle_displacement()
    #sample.calculate_shear_torque_2()
    sample.plot_torque_angle()
    sample.plot_shear_force_angle()

    print(sample)