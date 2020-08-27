AnisoDipFit
=========

Description
=========
The program AnisoDipFit was developed for the analysis of the Pulsed EPR Dipolar Spectroscopy (PDS) signals that correspond to spin systems consisting of one isotropic and one anisotropic S = 1/2 centers.

The program has three modes: simulation, fitting, and error analysis.

In the simulation mode, the PDS time trace / spectrum is simulated using the pre-defined geometric model of a spin system and the spectroscopic parameters of spin centers. 

In the fitting mode, the experimental PDS time trace / spectrum is fitted by means of genetic algorithm. In this case, the geometric model of a spin system is optimized until the simulated PDS time trace / spectrum provides the best fit to the experimental PDS time trace / spectrum.

In the error analysis mode, the errors of optimized fitting parameters are estimated.

Further description of the program can be found in the Manual and in the papers below.

General Information
=========
The source code of the program is written in Python 3.7 using the Python libraries numpy, scipy, matplotlib and libconf.

The Windows and Linux executables of the program are available at: 
https://github.com/dinarabdullin/AnisoDipFit/releases

By default, the program is a console application (i.e. it can be run from Terminal or Command Prompt). Alternatively, the Graphical User Interface of the program can be found at: 
https://github.com/PabloRauhCorro/AnisoDipFit (source code, manual, examples) and
https://github.com/PabloRauhCorro/AnisoDipFit/releases (executables, manual, examples: comming soon!).

Copyright
=========
This program can be distributed under GNU General Public License.

If you use this code please cite:
1) D. Abdullin, “AnisoDipFit: Simulation and Fitting of Pulsed EPR Dipolar Spectroscopy Data for Anisotropic Spin Centers”, Appl. Magn. Reson. 2020, 51, 725-748.
2) D. Abdullin, H. Matsuoka, M. Yulikov, N. Fleck, C. Klein, S. Spicher, G. Hagelueken, S. Grimme, A. Lützen, O. Schiemann, “Pulsed EPR Dipolar Spectroscopy under the Breakdown of the High-Field Approximation: The High-Spin Iron(III) Case”, Chem. Eur. J. 2019, 25, 8820-8828.
3) D. Abdullin, P. Brehm, N. Fleck, S. Spicher, S. Grimme, O. Schiemann, “Pulsed EPR Dipolar Spectroscopy on Spin Pairs with one Highly Anisotropic Spin Center: The Low-Spin Fe(III) Case”, Chem. Eur. J. 2019, 25, 14388-14398.
