import numpy as np

# # k = 4
# # For calculating the constants I should multiply uniform (x2) and VMF (x1 * x2) pdfs by to get a new valid pdf
# pdf_vmf = 0.0116640                  # value of pdf of VMF at edge / border with uniform
# int_probability_vmf = 0.0179862      # integration of VMF over hemisphere opposite to mean
# int_probability_uniform = 0.5        # integration of uniform over hemisphere
#
# x1 = 1 / (4 * np.pi * pdf_vmf)
# x2 = 1 / (int_probability_vmf * x1 + 0.5)
# print(x1)
# print(x2)
#
# print(int_probability_uniform * x2)
# print(int_probability_vmf * x1 * x2)

# k = 2
# For calculating the constants I should multiply uniform (x2) and VMF (x1 * x2) pdfs by to get a new valid pdf
pdf_vmf = 0.0438822907955184         # value of pdf of VMF at edge / border with uniform
int_probability_vmf = 0.119203      # integration of VMF over hemisphere opposite to mean
int_probability_uniform = 0.5        # integration of uniform over hemisphere

x1 = 1 / (4 * np.pi * pdf_vmf)
x2 = 1 / (int_probability_vmf * x1 + 0.5)
print(x1)
print(x2)

print(int_probability_uniform * x2)
print(int_probability_vmf * x1 * x2)