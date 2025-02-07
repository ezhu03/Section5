import sympy as sp

# Define symbols
z, beta, eps = sp.symbols('z beta eps', positive=True)
j, M = sp.symbols('j M', positive=True, integer=True)

# Define the product for the grand partition function
Xi = sp.Product(1 + z * sp.exp(-beta * j * eps), (j, 1, M)).doit()

# Display the result
sp.pretty_print(Xi)

