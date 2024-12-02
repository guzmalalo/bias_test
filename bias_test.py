import numpy as np
from numpy import typing
import matplotlib.pyplot as plt


class BiasTest:
    def __init__(
        self,
        displacement: typing.ArrayLike,
        force: typing.ArrayLike,
        width: float,
        length: float,
        thickness: float = 1.0,
        material_name: str = "Generic material",
    ):
        """
        Initialize the bias extension test.

        Parameters:
        -----------
        displacement :  array-like
            Experimental displacement (in mm).
        force :  array-like
            Experimental force (in N).
        width : float
            Initial width of the specimen (in mm).
        length : float
            Initial length of the specimen (in mm).
        thickness : float, optional
            Initial thickness of the specimen (in mm). Defaults to 1.0.
        material_name : str, optional
            Name of the material being tested. Defaults to 'Generic material'.

        Raises:
        -------
        ValueError
            If the length/width ratio is less than 2.0 or arrays have incompatible sizes.
        """

        # Experimental data
        try:
            self.displacement = np.asarray(displacement, dtype=float)
            self.force = np.asarray(force, dtype=float)
        except Exception as e:
            raise ValueError(f"Could not convert inputs to numeric arrays: {e}")

        # Validate array sizes
        if self.displacement.size != self.force.size:
            raise ValueError(
                f"Displacement and force arrays must have the same length. "
                f"Displacement: {self.displacement.size}, Force: {self.force.size}"
            )

        # Validate geometric constraints
        if length / width < 2.0:
            raise ValueError(
                f"Sample ratio (length/width) must be >= 2. Current ratio: {length/width:.2f}"
            )

        # Store experimental data and measurements
        self.number_mesures = self.displacement.size

        # Geometric properties
        self.name = material_name
        self.width = float(width)
        self.length = float(length)
        self.thickness = float(thickness)
        self.ratio = length / width

        # Useful geometric properties
        self.semi_sheared_area = width * width
        self.sheared_area = width * length - 1.5 * self.semi_sheared_area
        # .. diagonal of the virtual picture frame in central area
        self.diagonal_pf = length - width
        # .. length side of the virtual picture frame in central area
        self.length_side_pf = self.diagonal_pf / np.sqrt(2.0)

        # Angles
        self.inter_fiber_angle = self._calculate_inter_fibre_angle()
        self.shear_angle = self._calculate_shear_angle()

        # To compute
        self.shear_torque = np.zeros(self.number_mesures)
        self.shear_torque_computed = False
        self.shear_force = np.zeros(self.number_mesures)
        self.shear_force_computed = False

    def _calculate_inter_fibre_angle(self) -> float:
        return 2 * np.arccos((self.displacement + self.diagonal_pf) / (2.0 * self.length_side_pf))

    def _calculate_shear_angle(self) -> float:
        # Shear angle
        return np.pi / 2 - self.inter_fiber_angle
    
    def calculate_shear_torque(self):
        
        c = np.zeros(self.number_mesures)
        c[1] = 4*self.length_side_pf*self.force[1]*np.sin(self.inter_fiber_angle[1]/2.0)/(4*self.sheared_area - self.semi_sheared_area)

        for i in range(2,self.number_mesures):
            semi_angle = self.shear_angle[i]/2.0
            semi_torque = np.interp( semi_angle,self.shear_angle,c)

            c[i] = 1.0/self.semi_sheared_area * (self.length_side_pf *self.force[i]*np.sin(self.inter_fiber_angle[i]/2)- 0.5*self.semi_sheared_area*semi_torque)

        self.shear_torque = c
        self.shear_torque_computed = True

    def calculate_shear_torque_2(self):
        H = self.length
        W = self.width
        c = np.zeros(self.number_mesures)
        f = self.force
        sa = self.shear_angle

        # for i in range(2,self.number_mesures):
        #     semi_angle = sa[i]/2.0
        #     semi_torque = np.interp( semi_angle,sa,c)

        #     c[i] =((H/W-1)*f[i]*(np.cos(sa[i]/2) - np.sin(sa[i]/2)) - W*semi_torque)/(2*H-3*W)

        # self.shear_torque = c
        # self.shear_torque_computed = True

        self.shear_torque =  ((H/W-1)*self.force*(np.cos(self.shear_angle/2) - np.sin(self.shear_angle/2)))/(2*H-3*W)
        self.shear_torque_computed = True

    def calculate_shear_force(self):
        if not self.shear_torque_computed:
            self.calculate_shear_torque()

        self.shear_force = self.shear_torque / np.cos(self.shear_angle)
        self.shear_force_computed = True

    
    def plot_angle_displacement(self)->None:
        fig, ax = plt.subplots()

        ax.set_title(f'Shear angle  = f(displacement) : {self.name}')
        ax.plot(self.displacement, self.shear_angle*180/np.pi, 'k-', label='Theoretical')
        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Shear angle  (°)')

        ax.grid()
        plt.legend()
        plt.show()


    def plot_force_displacement(self)->None:
        fig, ax = plt.subplots()

        ax.set_title(f'Force = f(displacement) : {self.name}')
        ax.plot(self.displacement, self.force, 'k-', label='Experimental')
        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Force (N)')

        ax.grid()
        plt.legend()
        plt.show()


    def plot_torque_angle(self)->None:

        if not self.shear_torque_computed:
            self.calculate_shear_torque()

        fig, ax = plt.subplots()

        ax.set_title(f'Shear torque = f(shear angle) : {self.name}')
        ax.plot(self.shear_angle, self.shear_torque, 'k-', label='Experimental')
        ax.set_xlabel('Shear Angle (°)')
        ax.set_ylabel('Shear torque (N mm)')

        ax.grid()
        plt.legend()
        plt.show()


    def plot_shear_force_angle(self)->None:

        if not self.shear_force_computed:
            self.calculate_shear_force()

        fig, ax = plt.subplots()

        ax.set_title(f'Shear force = f(shear angle) : {self.name}')
        ax.plot(self.shear_angle*180/np.pi, self.shear_force, 'k-', label='Experimental')
        ax.set_xlabel('Shear Angle (°)')
        ax.set_ylabel('Shear force (N mm)')

        ax.grid()
        plt.legend()
        plt.show()

    def __str__(self):
        """
        Provides a formatted string representation of the bias extension properties.

        Returns:
            str: A human-readable description of the object's dimensions.
        """
        properties = [
            f"  Material:  {self.name}",
            f"  Width:     {self.width} mm",
            f"  Length:    {self.length} mm",
            f"  Thickness: {self.thickness} mm",
            f"  Sample ratio: {self.ratio}",
            f"  Sheared area: {self.sheared_area} mm^2",
            f"  Semi-sheared area: {self.semi_sheared_area} mm^2",
            f"  Experimental data: {self.number_mesures} points",
        ]

        return "Bias Extension Properties:\n" + "\n".join(properties)
