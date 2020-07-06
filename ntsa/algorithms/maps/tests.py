from numpy import mgrid, column_stack, array

from ntsa.algorithms.maps import LyapunovExponents
from ntsa.algorithms.maps import Henon, Baker


henon_le = LyapunovExponents(map=Henon())
henon_solution = henon_le.calculate_from_initial_conditions(array([-0.5, 0.25]))
print(henon_solution)


set =  array([[-1., -1.], [-1., -0.75], [-1., -0.5], [-1., -0.25], [-1., 0.], [-1., 0.25], [-1., 0.5], [-1., 0.75], [-0.75, -1.], [-0.75, -0.75], [-0.75, -0.5], [-0.75, -0.25], [-0.75, 0.], [-0.75, 0.25], [-0.75, 0.5], [-0.75, 0.75], [-0.5, -1.], [-0.5, -0.75], [-0.5, -0.5], [-0.5, -0.25], [-0.5, 0.], [-0.5, 0.25], [-0.5, 0.5], [-0.5, 0.75], [-0.25, -1.], [-0.25, -0.75], [-0.25, -0.5], [-0.25, -0.25], [-0.25, 0.], [-0.25, 0.25], [-0.25, 0.5], [-0.25, 0.75], [0., -1.], [0., -0.75], [0., -0.5], [0., -0.25], [0., 0.], [0., 0.25], [0., 0.5]])

henon_le = LyapunovExponents(map=Henon())
henon_solution = henon_le.calculate_over_set_of_initial_conditions(set)
print(henon_solution)



baker_le = LyapunovExponents(map=Baker())
baker_solution = baker_le.calculate_from_initial_conditions(array([0.75, 0.75]))
print(baker_solution)
