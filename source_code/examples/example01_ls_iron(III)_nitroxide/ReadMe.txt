NAME
--------------------------------------
Example 1


DESCRIPTION
--------------------------------------
Spectral simulations for the spin system consisting of the low-spin Fe(III) ion and the nitroxide.


HOW TO RUN  
--------------------------------------
1) Open Terminal or Command Prompt.

2) Navigate to the root directory of the program:
   cd …/AnisoDipFit/source_code
   
3) Run the program by the following command:
   python main.py examples/example01_ls_iron(III)_nitroxide/config_ex01_1.cfg (simulates the dipolar spectrum vs xi, phi = 0°)
   python main.py examples/example01_ls_iron(III)_nitroxide/config_ex01_2.cfg (simulates the dipolar spectrum vs xi, phi = 30°)
   python main.py examples/example01_ls_iron(III)_nitroxide/config_ex01_3.cfg (simulates the dipolar spectrum vs xi, phi = 60°)
   python main.py examples/example01_ls_iron(III)_nitroxide/config_ex01_4.cfg (simulates the dipolar spectrum vs xi, phi = 90°)

4) The results of the AnisoDipFit analysis will be saved in the folder:
   AnisoDipFit/source_code/examples/example01_ls_iron(III)_nitroxide/
